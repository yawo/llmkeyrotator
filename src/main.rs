use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use csv::ReaderBuilder;
use futures::StreamExt;
use tokio_stream::StreamExt as TokioStreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    env,
    fs::File,
    io::BufReader,
    net::SocketAddr,
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

mod anthropic {
    use serde_json::Value;

    /// Inject Anthropic cache_control markers into the body (in-place).
    /// - System prompt → wrapped as array block with cache_control
    /// - Last user message → last content block gets cache_control
    /// Safe to call on already-hinted bodies (idempotent for the system block).
    pub fn apply_cache_hints(body: &mut Value) {
        let cc = serde_json::json!({"type": "ephemeral"});

        // System: string → array block with cache_control
        if let Some(system) = body.get("system") {
            let new_system = if let Some(s) = system.as_str() {
                if !s.is_empty() {
                    Some(serde_json::json!([{"type": "text", "text": s, "cache_control": cc}]))
                } else {
                    None
                }
            } else if system.is_array() {
                let mut arr = system.as_array().unwrap().clone();
                if let Some(last) = arr.last_mut() {
                    if last.get("cache_control").is_none() {
                        last["cache_control"] = cc.clone();
                    }
                }
                Some(Value::Array(arr))
            } else {
                None
            };
            if let Some(s) = new_system {
                body["system"] = s;
            }
        }

        // Last user message: add cache_control to its last content block
        if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            if let Some(last_user) = messages.iter_mut().rev().find(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
            }) {
                match last_user.get_mut("content") {
                    Some(Value::Array(blocks)) => {
                        if let Some(last_block) = blocks.last_mut() {
                            if last_block.get("cache_control").is_none() {
                                last_block["cache_control"] = cc;
                            }
                        }
                    }
                    Some(Value::String(s)) => {
                        let text = s.clone();
                        last_user["content"] = serde_json::json!([{
                            "type": "text", "text": text, "cache_control": cc
                        }]);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn request_to_openai(body: &mut Value) -> Value {
        let mut openai = serde_json::Map::new();

        if let Some(model) = body.get("model") {
            openai.insert("model".to_string(), model.clone());
        }

        let mut messages = Vec::new();

        if let Some(system) = body.get("system") {
            if system.is_string() && !system.as_str().unwrap_or("").is_empty() {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": system
                }));
            } else if system.is_array() {
                let texts: Vec<String> = system
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                    .collect();
                if !texts.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "system",
                        "content": texts.join("\n")
                    }));
                }
            }
        }

        if let Some(anthropic_messages) = body.get("messages").and_then(|m| m.as_array()) {
            for msg in anthropic_messages {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                let content = msg.get("content");

                match content {
                    Some(Value::String(s)) => {
                        messages.push(serde_json::json!({
                            "role": role,
                            "content": s.clone()
                        }));
                    }
                    Some(Value::Array(blocks)) => {
                        let mut text_parts: Vec<String> = Vec::new();
                        let mut tool_calls: Vec<Value> = Vec::new();
                        let mut tool_results: Vec<Value> = Vec::new();

                        for block in blocks {
                            let btype = match block.get("type").and_then(|t| t.as_str()) {
                                Some(t) => t,
                                None => continue,
                            };
                            match btype {
                                "text" => {
                                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                        text_parts.push(text.to_string());
                                    }
                                }
                                "tool_use" => {
                                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let input = block.get("input").unwrap_or(&Value::Null);
                                    let arguments = serde_json::to_string(input).unwrap_or_default();
                                    tool_calls.push(serde_json::json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments
                                        }
                                    }));
                                }
                                "tool_result" => {
                                    let tool_use_id = block.get("tool_use_id").and_then(|t| t.as_str()).unwrap_or("");
                                    let result_content = block.get("content").map(|c| {
                                        if c.is_string() {
                                            c.as_str().unwrap_or("").to_string()
                                        } else {
                                            serde_json::to_string(c).unwrap_or_default()
                                        }
                                    }).unwrap_or_default();
                                    tool_results.push(serde_json::json!({
                                        "role": "tool",
                                        "tool_call_id": tool_use_id,
                                        "content": result_content
                                    }));
                                }
                                _ => {}
                            }
                        }

                        if !tool_calls.is_empty() || !text_parts.is_empty() {
                            let mut msg_obj = serde_json::Map::new();
                            msg_obj.insert("role".to_string(), Value::String(role.to_string()));

                            if !text_parts.is_empty() {
                                msg_obj.insert("content".to_string(), Value::String(text_parts.join("\n")));
                            } else if !tool_calls.is_empty() {
                                msg_obj.insert("content".to_string(), Value::Null);
                            }

                            if !tool_calls.is_empty() {
                                msg_obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
                            }

                            messages.push(Value::Object(msg_obj));
                        }

                        for tr in tool_results {
                            messages.push(tr);
                        }
                    }
                    _ => {
                        messages.push(serde_json::json!({
                            "role": role,
                            "content": Value::Null
                        }));
                    }
                }
            }
        }

        openai.insert("messages".to_string(), Value::Array(messages));

        if let Some(max_tokens) = body.get("max_tokens") {
            openai.insert("max_tokens".to_string(), max_tokens.clone());
        }

        if let Some(temperature) = body.get("temperature") {
            openai.insert("temperature".to_string(), temperature.clone());
        }

        if let Some(top_p) = body.get("top_p") {
            openai.insert("top_p".to_string(), top_p.clone());
        }

        if let Some(stop) = body.get("stop_sequences") {
            openai.insert("stop".to_string(), stop.clone());
        }

        if let Some(stream) = body.get("stream") {
            openai.insert("stream".to_string(), stream.clone());
        }

        let has_tools = body
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        if has_tools {
            let converted_tools: Vec<Value> = body
                .get("tools")
                .and_then(|t| t.as_array())
                .unwrap()
                .iter()
                .map(|tool| {
                    let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
                    let parameters = tool.get("input_schema").cloned().unwrap_or(Value::Null);
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters
                        }
                    })
                })
                .collect();
            openai.insert("tools".to_string(), Value::Array(converted_tools));

            if let Some(tool_choice) = body.get("tool_choice") {
                let converted = match tool_choice.get("type").and_then(|t| t.as_str()) {
                    Some("auto") => serde_json::json!("auto"),
                    Some("any") => serde_json::json!("required"),
                    Some("tool") => {
                        if let Some(name) = tool_choice.get("name") {
                            serde_json::json!({
                                "type": "function",
                                "function": { "name": name }
                            })
                        } else {
                            serde_json::json!("auto")
                        }
                    }
                    _ => serde_json::json!("auto"),
                };
                openai.insert("tool_choice".to_string(), converted);
            }
        }

        Value::Object(openai)
    }

    pub fn response_to_anthropic(openai: &Value) -> Value {
        let id = openai
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let model = openai
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let choice = openai
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());

        let mut content = Vec::new();
        let mut stop_reason = "end_turn";

        if let Some(choice) = choice {
            if let Some(msg) = choice.get("message") {
                if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                    if !text.is_empty() {
                        content.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                }

                if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tool_calls {
                        let tc_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        let tc_name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("");
                        let tc_args = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}");
                        let input: Value = serde_json::from_str(tc_args).unwrap_or_default();
                        content.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": input
                        }));
                    }
                    stop_reason = "tool_use";
                }
            }

            if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                if stop_reason != "tool_use" {
                    stop_reason = match reason {
                        "stop" => "end_turn",
                        "length" => "max_tokens",
                        "tool_calls" => "tool_use",
                        _ => "end_turn",
                    };
                }
            }
        }

        let usage = openai.get("usage").and_then(|u| u.as_object());
        let input_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        serde_json::json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        })
    }

    pub fn sse_message_start(id: &str, model: &str) -> (String, String) {
        (
            "message_start".to_string(),
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            }).to_string(),
        )
    }

    pub fn sse_chunk_to_anthropic(openai_chunk: &str, has_text: &mut bool, opened_tool_indices: &mut std::collections::HashSet<usize>) -> Vec<(String, String)> {
        let mut events = Vec::new();

        let chunk: Value = match serde_json::from_str(openai_chunk) {
            Ok(v) => v,
            Err(_) => return events,
        };

        let mut stop_reason: Option<String> = None;

        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            if !*has_text {
                                events.push((
                                    "content_block_start".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": { "type": "text", "text": "" }
                                    }).to_string(),
                                ));
                                *has_text = true;
                            }
                            events.push((
                                "content_block_delta".to_string(),
                                serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": text }
                                }).to_string(),
                            ));
                        }
                    }

                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                        for (i, tc) in tool_calls.iter().enumerate() {
                            let tc_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(i as u64) as usize;
                            // Offset by 1 so tool blocks never collide with the text block at index 0
                            let anthropic_index = tc_index + 1;
                            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let tc_name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("");
                            let tc_args = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .unwrap_or("");

                            if !opened_tool_indices.contains(&anthropic_index) {
                                events.push((
                                    "content_block_start".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_start",
                                        "index": anthropic_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tc_id,
                                            "name": tc_name,
                                            "input": {}
                                        }
                                    }).to_string(),
                                ));
                                opened_tool_indices.insert(anthropic_index);
                            }

                            if !tc_args.is_empty() {
                                events.push((
                                    "content_block_delta".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_delta",
                                        "index": anthropic_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": tc_args
                                        }
                                    }).to_string(),
                                ));
                            }
                        }
                    }
                }

                if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                    stop_reason = Some(match reason {
                        "stop" => "end_turn".to_string(),
                        "length" => "max_tokens".to_string(),
                        "tool_calls" => "tool_use".to_string(),
                        _ => "end_turn".to_string(),
                    });
                }
            }
        }

        if let Some(reason) = stop_reason {
            let mut all_stop_indices: Vec<usize> = Vec::new();
            if *has_text {
                all_stop_indices.push(0);
            }
            all_stop_indices.extend(opened_tool_indices.iter().copied());
            all_stop_indices.sort();
            all_stop_indices.dedup();
            for idx in all_stop_indices {
                events.push((
                    "content_block_stop".to_string(),
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": idx
                    }).to_string(),
                ));
            }

            events.push((
                "message_delta".to_string(),
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": reason, "stop_sequence": null },
                    "usage": { "output_tokens": 0 }
                }).to_string(),
            ));

            events.push((
                "message_stop".to_string(),
                serde_json::json!({ "type": "message_stop" }).to_string(),
            ));
        }

        events
    }
}

mod dns {
    use reqwest::dns::{Addrs, Name, Resolve, Resolving};
    use std::net::SocketAddr;
    use trust_dns_resolver::config::{NameServerConfig, Protocol, ResolverConfig, ResolverOpts};
    use trust_dns_resolver::TokioAsyncResolver;

    pub struct GoogleDnsResolver {
        inner: TokioAsyncResolver,
    }

    impl GoogleDnsResolver {
        pub fn new() -> Self {
            let mut config = ResolverConfig::new();
            config.add_name_server(NameServerConfig {
                socket_addr: "8.8.8.8:53".parse::<SocketAddr>().unwrap(),
                protocol: Protocol::Udp,
                tls_dns_name: None,
                trust_negative_responses: false,
                bind_addr: None,
            });
            config.add_name_server(NameServerConfig {
                socket_addr: "8.8.4.4:53".parse::<SocketAddr>().unwrap(),
                protocol: Protocol::Udp,
                tls_dns_name: None,
                trust_negative_responses: false,
                bind_addr: None,
            });
            config.add_name_server(NameServerConfig {
                socket_addr: "1.1.1.1:53".parse::<SocketAddr>().unwrap(),
                protocol: Protocol::Udp,
                tls_dns_name: None,
                trust_negative_responses: false,
                bind_addr: None,
            });
            let resolver = TokioAsyncResolver::tokio(config, ResolverOpts::default());
            Self { inner: resolver }
        }
    }

    impl Resolve for GoogleDnsResolver {
        fn resolve(&self, name: Name) -> Resolving {
            let inner = self.inner.clone();
            let host = name.as_str().to_string();
            Box::pin(async move {
                let response = inner.lookup_ip(host).await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
                let addrs: Vec<SocketAddr> = response
                    .iter()
                    .map(|ip| SocketAddr::new(ip, 0))
                    .collect();
                Ok(Box::new(addrs.into_iter()) as Addrs)
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Provider {
    name: String,
    base_url: String,
    model: String,
    api_key: String,
}

#[derive(Debug)]
struct Stats {
    total_requests: AtomicUsize,
    total_errors: AtomicUsize,
    total_rotations: AtomicUsize,
    provider_failures: Vec<AtomicUsize>,
    skipped_providers: Arc<Mutex<std::collections::HashSet<String>>>,
}

impl Stats {
    fn new(provider_count: usize) -> Self {
        let mut provider_failures = Vec::with_capacity(provider_count);
        for _ in 0..provider_count {
            provider_failures.push(AtomicUsize::new(0));
        }
        Self {
            total_requests: AtomicUsize::new(0),
            total_errors: AtomicUsize::new(0),
            total_rotations: AtomicUsize::new(0),
            provider_failures,
            skipped_providers: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }
}

#[derive(Clone)]
struct AppState {
    providers: Arc<Vec<Provider>>,
    current_index: Arc<AtomicUsize>,
    client: reqwest::Client,
    stats: Arc<Stats>,
    expected_api_key: String,
    cache_enabled: bool,
    compress_enabled: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

impl AppState {
    fn record_failure(&self, provider_name: &str) {
        if let Some(idx) = self.providers.iter().position(|p| p.name == provider_name) {
            self.stats.provider_failures[idx].fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn load_providers(csv_path: &str) -> Vec<Provider> {
    let file = File::open(csv_path).expect("Failed to open CSV file");
    let reader = BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);
    let mut providers = Vec::new();
    for result in csv_reader.deserialize() {
        let provider: Provider = result.expect("Failed to parse CSV row");
        providers.push(provider);
    }
    info!(count = providers.len(), "Loaded providers from CSV");
    providers
}

async fn skip_provider_permanently(state: &AppState, name: &str) {
    let mut skipped = state.stats.skipped_providers.lock().await;
    if skipped.insert(name.to_string()) {
        warn!(provider = %name, "Provider permanently skipped (payload too large)");
        // Advance index if current provider is the one being skipped
        let n = state.providers.len();
        let idx = state.current_index.load(Ordering::Relaxed);
        if state.providers[idx % n].name == name {
            state.current_index.fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn get_next_available_provider(state: &AppState) -> Option<&Provider> {
    let skipped = state.stats.skipped_providers.lock().await;
    let n = state.providers.len();
    let start = state.current_index.load(Ordering::Relaxed);
    for i in 0..n {
        let p = &state.providers[(start + i) % n];
        if !skipped.contains(&p.name) {
            return Some(p);
        }
    }
    None
}

async fn rotate_provider(state: &AppState) {
    let idx = state.current_index.fetch_add(1, Ordering::Relaxed);
    let new_idx = idx + 1;
    let from_name = &state.providers[idx % state.providers.len()].name;
    let to_name = &state.providers[new_idx % state.providers.len()].name;
    warn!(from = from_name, to = to_name, "Rotating to next provider");
    state.stats.total_rotations.fetch_add(1, Ordering::Relaxed);
}

fn auth_check(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    if state.expected_api_key.is_empty() {
        return Ok(());
    }
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth.strip_prefix("Bearer ").unwrap_or(auth);
    if token != state.expected_api_key {
        warn!("Authentication failed");
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Unauthorized".to_string(),
            }),
        )
            .into_response());
    }
    Ok(())
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let provider_name = get_next_available_provider(&state)
        .await
        .map(|p| p.name.as_str())
        .unwrap_or("none");
    Json(serde_json::json!({
        "status": "ok",
        "providers": state.providers.len(),
        "current_provider": provider_name,
        "total_requests": state.stats.total_requests.load(Ordering::Relaxed),
        "total_errors": state.stats.total_errors.load(Ordering::Relaxed),
        "total_rotations": state.stats.total_rotations.load(Ordering::Relaxed),
    }))
}

async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let failures: std::collections::HashMap<&str, usize> = state
        .providers
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.as_str(), state.stats.provider_failures[i].load(Ordering::Relaxed)))
        .collect();
    Json(serde_json::json!({
        "total_requests": state.stats.total_requests.load(Ordering::Relaxed),
        "total_errors": state.stats.total_errors.load(Ordering::Relaxed),
        "total_rotations": state.stats.total_rotations.load(Ordering::Relaxed),
        "provider_failures": failures,
    }))
}

const COMPRESS_PROMPT: &str = "Ultra-compressed replies. Drop articles/fillers/pleasantries/hedging. Fragments OK. Short synonyms. Technical terms exact. Code blocks unchanged. Errors quoted exact. Pattern: [thing] [action] [reason]. Abbreviate prose (DB/auth/config/req/res/fn/impl), arrows for causality (X→Y). Never abbreviate code symbols/fn names/API names/error strings.";

/// Append compress hint to system prompt in an OpenAI-format body (mutates in place).
fn apply_compress_hint_openai(body: &mut Value) {
    if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        if let Some(sys) = messages.iter_mut().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system")) {
            let existing = sys["content"].as_str().unwrap_or("").to_string();
            if !existing.contains(COMPRESS_PROMPT) {
                sys["content"] = Value::String(format!("{}\n\n{}", existing.trim_end(), COMPRESS_PROMPT));
            }
        } else {
            messages.insert(0, serde_json::json!({"role": "system", "content": COMPRESS_PROMPT}));
        }
    }
}

/// Append compress hint to system prompt in an Anthropic-format body (mutates in place).
fn apply_compress_hint_anthropic(body: &mut Value) {
    match body.get("system") {
        Some(Value::String(s)) => {
            if !s.contains(COMPRESS_PROMPT) {
                let s = format!("{}\n\n{}", s.trim_end(), COMPRESS_PROMPT);
                body["system"] = Value::String(s);
            }
        }
        Some(Value::Array(_)) => {
            if let Some(arr) = body["system"].as_array_mut() {
                if !arr.iter().any(|b| b.get("text").and_then(|t| t.as_str()) == Some(COMPRESS_PROMPT)) {
                    arr.push(serde_json::json!({"type": "text", "text": COMPRESS_PROMPT}));
                }
            }
        }
        _ => {
            body["system"] = Value::String(COMPRESS_PROMPT.to_string());
        }
    }
}

fn is_streaming_request(body: &Value) -> bool {
    body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
}

async fn proxy_openai(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(resp) = auth_check(&state, &headers) {
        return resp;
    }

    state.stats.total_requests.fetch_add(1, Ordering::Relaxed);

    let streaming = is_streaming_request(&body);
    let max_retries = state.providers.len();

    if state.compress_enabled {
        apply_compress_hint_openai(&mut body);
    }

    for attempt in 0..max_retries {
        let provider = match get_next_available_provider(&state).await {
            Some(p) => p,
            None => break,
        };
        body["model"] = Value::String(provider.model.clone());

        let start = std::time::Instant::now();
        info!(
            attempt = attempt + 1,
            provider = %provider.name,
            model = %provider.model,
            streaming = streaming,
            "Forwarding request"
        );

        let path = uri.path();
        let path = path
            .strip_prefix("/v1")
            .unwrap_or(path);
        let url = format!("{}{}", provider.base_url.trim_end_matches('/'), path);
        let request_timeout: u64 = env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(120);

        let request_builder = state
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        let send_result = tokio::time::timeout(
            std::time::Duration::from_secs(request_timeout),
            request_builder.send(),
        ).await;
        let send_result = match send_result {
            Ok(r) => r,
            Err(_) => {
                error!(provider = %provider.name, request_timeout=%request_timeout, "OpenAI request timed out after 120s");
                state.stats.total_errors.fetch_add(1, Ordering::Relaxed);
                state.record_failure(&provider.name);
                if attempt < max_retries - 1 {
                    warn!(failed_provider = %provider.name, next_attempt = attempt + 2, "Provider timed out, retrying with next");
                    rotate_provider(&state).await;
                }
                continue;
            }
        };
        match send_result {
            Ok(resp) => {
                let status = resp.status();
                let ttfb = start.elapsed();
                if status.is_success() {
                    info!(provider = %provider.name, ttfb_ms = ttfb.as_millis(), "Response received");
                    if streaming {
                        let buf = Arc::new(std::sync::Mutex::new(String::new()));
                        let timed_stream = TokioStreamExt::timeout(resp.bytes_stream(), std::time::Duration::from_secs(30));
                        let stream = futures::StreamExt::map(timed_stream, move |result| {
                            match result {
                                Ok(Ok(bytes)) => {
                                    let mut buf = buf.lock().unwrap();
                                    buf.push_str(&String::from_utf8_lossy(&bytes));
                                    let mut events = Vec::new();
                                    while let Some(pos) = buf.find('\n') {
                                        let line = buf[..pos].trim().to_string();
                                        buf.drain(..=pos);
                                        if line.starts_with("data: ") {
                                            let data = &line[6..];
                                            if data != "[DONE]" {
                                                events.push(Ok(
                                                    axum::response::sse::Event::default()
                                                        .data(data.to_string()),
                                                ));
                                            }
                                        }
                                    }
                                    futures::stream::iter(events)
                                }
                                Ok(Err(e)) => {
                                    error!(error = %e, "Stream error");
                                    futures::stream::iter(vec![Err(std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        e.to_string(),
                                    ))])
                                }
                                Err(_) => {
                                    error!("Stream chunk timed out after 30s");
                                    futures::stream::iter(vec![Err(std::io::Error::new(
                                        std::io::ErrorKind::TimedOut,
                                        "stream chunk timeout",
                                    ))])
                                }
                            }
                        });
                        let flat_stream = stream.flatten();
                        return Sse::new(flat_stream)
                            .keep_alive(
                                axum::response::sse::KeepAlive::new()
                                    .interval(std::time::Duration::from_secs(15))
                                    .text("keep-alive"),
                            )
                            .into_response();
                    } else {
                        match resp.json::<Value>().await {
                            Ok(json) => return Json(json).into_response(),
                            Err(e) => {
                                error!(error = %e, provider = %provider.name, "Failed to parse response");
                            }
                        }
                    }
                } else {
                    let err_body = resp.text().await.unwrap_or_default();
                    error!(
                        status = %status,
                        provider = %provider.name,
                        body = %err_body,
                        ttfb_ms = ttfb.as_millis(),
                        "Provider returned error"
                    );
                    if status == StatusCode::PAYLOAD_TOO_LARGE {
                        skip_provider_permanently(&state, &provider.name).await;
                    }
                }
            }
            Err(e) => {
                let elapsed = start.elapsed();
                error!(error = %e, provider = %provider.name, elapsed_ms = elapsed.as_millis(), "Request failed");
            }
        }

        state.stats.total_errors.fetch_add(1, Ordering::Relaxed);
        state.record_failure(&provider.name);

        if attempt < max_retries - 1 {
            warn!(
                failed_provider = %provider.name,
                next_attempt = attempt + 2,
                max_retries = max_retries,
                "Provider failed, retrying with next"
            );
            rotate_provider(&state).await;
        }
    }

    error!("All providers exhausted for OpenAI request");
    (
        StatusCode::BAD_GATEWAY,
        Json(ErrorResponse {
            error: "All providers failed".to_string(),
        }),
    )
        .into_response()
}

async fn proxy_anthropic(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(resp) = auth_check(&state, &headers) {
        return resp;
    }

    state.stats.total_requests.fetch_add(1, Ordering::Relaxed);

    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("anthropic-request-id"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    info!(path = %uri.path(), request_id = %request_id, "Anthropic request received");

    let streaming = is_streaming_request(&body);
    let max_retries = state.providers.len();
    let request_timeout: u64 = env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(120);


    

    // Prepare cached and uncached OpenAI bodies up front
    if state.compress_enabled {
        apply_compress_hint_anthropic(&mut body);
    }
    let openai_body_plain = anthropic::request_to_openai(&mut body);
    let openai_body_cached = if state.cache_enabled {
        let mut cached = body.clone();
        anthropic::apply_cache_hints(&mut cached);
        Some(anthropic::request_to_openai(&mut cached))
    } else {
        None
    };

    for attempt in 0..max_retries {
        let provider = match get_next_available_provider(&state).await {
            Some(p) => p,
            None => break,
        };

        // Try cached body first (if enabled), fall back to plain on 400
        let candidates: &[(&Value, bool)] = if let Some(ref cached) = openai_body_cached {
            &[(cached, true), (&openai_body_plain, false)]
        } else {
            &[(&openai_body_plain, false)]
        };

        'inner: for &(base_body, with_cache) in candidates {
            let mut forward_body = base_body.clone();
            forward_body["model"] = Value::String(provider.model.clone());

            let start = std::time::Instant::now();
            info!(
                attempt = attempt + 1,
                provider = %provider.name,
                model = %provider.model,
                streaming = streaming,
                cache = with_cache,
                "Forwarding Anthropic request"
            );

            let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));

            let request_builder = state
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", provider.api_key))
                .header("Content-Type", "application/json")
                .json(&forward_body);

            let send_result = tokio::time::timeout(
                std::time::Duration::from_secs(request_timeout),
                request_builder.send(),
            ).await;
            let send_result = match send_result {
                Ok(r) => r,
                Err(_) => {
                    error!(provider = %provider.name,request_timeout = %request_timeout,  "Anthropic request timed out after 120s");
                    state.stats.total_errors.fetch_add(1, Ordering::Relaxed);
                    state.record_failure(&provider.name);
                    if attempt < max_retries - 1 {
                        warn!(failed_provider = %provider.name, next_attempt = attempt + 2, "Provider timed out, retrying with next");
                        rotate_provider(&state).await;
                    }
                    break 'inner;
                }
            };
            match send_result {
                Ok(resp) => {
                    let status = resp.status();
                    let ttfb = start.elapsed();
                    if status.is_success() {
                        info!(provider = %provider.name, ttfb_ms = ttfb.as_millis(), "Response received");
                        if streaming {
                            // Use a channel so we can return HTTP 200 + SSE headers immediately,
                            // before the upstream sends any bytes. This prevents Claude Code from
                            // retrying the request when TTFB is slow.
                            let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(32);

                            // Send an immediate ping so the HTTP response flushes to the client right away.
                            let ping = b"event: ping\ndata: {\"type\":\"ping\"}\n\n";
                            let _ = tx.send(Ok(axum::body::Bytes::from_static(ping))).await;

                            tokio::spawn(async move {
                                let mut buf = String::new();
                                let mut sent_start = false;
                                let mut has_text = false;
                                let mut opened_tools: std::collections::HashSet<usize> = std::collections::HashSet::new();

                                let timed_stream = TokioStreamExt::timeout(resp.bytes_stream(), std::time::Duration::from_secs(30));
                                tokio::pin!(timed_stream);
                                while let Some(result) = TokioStreamExt::next(&mut timed_stream).await {
                                    match result {
                                        Ok(chunk_result) => match chunk_result {
                                            Ok(bytes) => {
                                                buf.push_str(&String::from_utf8_lossy(&bytes));
                                                while let Some(pos) = buf.find('\n') {
                                                    let line = buf[..pos].trim().to_string();
                                                    buf.drain(..=pos);
                                                    if !line.starts_with("data: ") { continue; }
                                                    let data = &line[6..];
                                                    if data == "[DONE]" { continue; }
                                                    if !sent_start {
                                                        sent_start = true;
                                                        let (id, model) = if let Ok(v) = serde_json::from_str::<Value>(data) {
                                                            let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                            let model = v.get("model").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                            (id, model)
                                                        } else {
                                                            (String::new(), String::new())
                                                        };
                                                        let (et, ed) = anthropic::sse_message_start(&id, &model);
                                                        let msg = format!("event: {}\ndata: {}\n\n", et, ed);
                                                        if tx.send(Ok(axum::body::Bytes::from(msg))).await.is_err() { return; }
                                                    }
                                                    for (event_type, event_data) in anthropic::sse_chunk_to_anthropic(data, &mut has_text, &mut opened_tools) {
                                                        let msg = format!("event: {}\ndata: {}\n\n", event_type, event_data);
                                                        if tx.send(Ok(axum::body::Bytes::from(msg))).await.is_err() { return; }
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                error!(error = %e, "Anthropic stream error");
                                                let _ = tx.send(Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))).await;
                                                return;
                                            }
                                        },
                                        Err(_) => {
                                            error!("Anthropic stream chunk timed out after 30s");
                                            let _ = tx.send(Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "stream chunk timeout"))).await;
                                            return;
                                        }
                                    }
                                }
                            });

                            let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
                            return Response::builder()
                                .status(StatusCode::OK)
                                .header("Content-Type", "text/event-stream")
                                .header("Cache-Control", "no-cache")
                                .header("Connection", "keep-alive")
                                .body(body)
                                .unwrap();
                        } else {
                            match resp.json::<Value>().await {
                                Ok(json) => {
                                    let anthropic_resp = anthropic::response_to_anthropic(&json);
                                    return Json(anthropic_resp).into_response();
                                }
                                Err(e) => {
                                    error!(error = %e, provider = %provider.name, "Failed to parse Anthropic response");
                                }
                            }
                        }
                    } else if status == StatusCode::BAD_REQUEST && with_cache {
                        // Provider rejected cache_control — retry same provider without cache
                        let err_body = resp.text().await.unwrap_or_default();
                        warn!(provider = %provider.name, body = %err_body, "Cache hints rejected (400), retrying without cache");
                        continue; // try plain body next
                    } else {
                        let err_body = resp.text().await.unwrap_or_default();
                        error!(
                            status = %status,
                            provider = %provider.name,
                            body = %err_body,
                            ttfb_ms = ttfb.as_millis(),
                            "Provider returned error for Anthropic request"
                        );
                        if status == StatusCode::PAYLOAD_TOO_LARGE {
                            skip_provider_permanently(&state, &provider.name).await;
                        }
                    }
                }
                Err(e) => {
                    let elapsed = start.elapsed();
                    error!(error = %e, provider = %provider.name, elapsed_ms = elapsed.as_millis(), "Anthropic request failed");
                }
            }
            break 'inner;
        }

        state.stats.total_errors.fetch_add(1, Ordering::Relaxed);
        state.record_failure(&provider.name);

        if attempt < max_retries - 1 {
            warn!(
                failed_provider = %provider.name,
                next_attempt = attempt + 2,
                max_retries = max_retries,
                "Provider failed, retrying with next"
            );
            rotate_provider(&state).await;
        }
    }

    error!("All providers exhausted for Anthropic request");
    (
        StatusCode::BAD_GATEWAY,
        Json(ErrorResponse {
            error: "All providers failed".to_string(),
        }),
    )
        .into_response()
}

async fn catch_all(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    method: axum::http::Method,
    body: Body,
) -> Response {
    if let Err(resp) = auth_check(&state, &headers) {
        return resp;
    }

    state.stats.total_requests.fetch_add(1, Ordering::Relaxed);

    let path = uri.path();
    let path = path
        .strip_prefix("/v1")
        .or_else(|| path.strip_prefix("/anthropic"))
        .unwrap_or(path);

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    let raw_bytes = match axum::body::to_bytes(body, 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "Failed to read body");
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse { error: "Request body too large".to_string() }),
            ).into_response();
        }
    };

    let max_retries = state.providers.len();
    // Parse JSON once outside the loop; model field is overwritten per-attempt
    let parsed_json: Option<Value> = serde_json::from_slice(&raw_bytes).ok();

    for attempt in 0..max_retries {
        let provider = match get_next_available_provider(&state).await {
            Some(p) => p,
            None => break,
        };

        let body_bytes = if let Some(mut json) = parsed_json.clone() {
            json["model"] = Value::String(provider.model.clone());
            serde_json::to_vec(&json).unwrap_or(raw_bytes.to_vec())
        } else {
            raw_bytes.to_vec()
        };

        let url = format!("{}{}", provider.base_url.trim_end_matches('/'), path);
        let start = std::time::Instant::now();
        let request_timeout: u64 = env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(120);


        info!(attempt = attempt + 1, provider = %provider.name, path = path, url = %url, "Proxying request");

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(request_timeout),
            state.client
                .request(method.clone(), &url)
                .header("Authorization", format!("Bearer {}", provider.api_key))
                .header("Content-Type", &content_type)
                .body(body_bytes)
                .send(),
        ).await;

        match result {
            Ok(Ok(resp)) => {
                let ttfb = start.elapsed();
                let status = resp.status();
                info!(provider = %provider.name, status = %status, ttfb_ms = ttfb.as_millis(), "Proxy response");
                if status.is_success() || status.is_redirection() {
                    let resp_headers = resp.headers().clone();
                    match resp.bytes().await {
                        Ok(bytes) => {
                            let mut builder = Response::builder().status(status);
                            for (key, value) in resp_headers.iter() {
                                if key != "transfer-encoding" && key != "content-length" {
                                    builder = builder.header(key, value);
                                }
                            }
                            return builder.body(Body::from(bytes)).unwrap_or_else(|e| {
                                error!(error = %e, "Failed to build response");
                                StatusCode::INTERNAL_SERVER_ERROR.into_response()
                            });
                        }
                        Err(e) => {
                            error!(error = %e, "Failed to read response body");
                        }
                    }
                } else {
                    let err_body = resp.text().await.unwrap_or_default();
                    error!(status = %status, provider = %provider.name, body = %err_body, "Catch-all provider error");
                    if status == StatusCode::PAYLOAD_TOO_LARGE {
                        skip_provider_permanently(&state, &provider.name).await;
                    }
                }
            }
            Ok(Err(e)) => {
                let elapsed = start.elapsed();
                error!(error = %e, provider = %provider.name, elapsed_ms = elapsed.as_millis(), "Catch-all request failed");
            }
            Err(_) => {
                error!(provider = %provider.name, request_timeout=%request_timeout, "Catch-all request timed out after 120s");
            }
        }

        state.stats.total_errors.fetch_add(1, Ordering::Relaxed);
        state.record_failure(&provider.name);
        if attempt < max_retries - 1 {
            warn!(failed_provider = %provider.name, next_attempt = attempt + 2, "Catch-all provider failed, retrying");
            rotate_provider(&state).await;
        }
    }

    error!("All providers exhausted for catch-all request");
    (
        StatusCode::BAD_GATEWAY,
        Json(ErrorResponse { error: "All providers failed".to_string() }),
    ).into_response()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG").unwrap_or_else(|_| "llmkeyrotator=info".to_string()),
        )
        .init();

    let csv_path = env::var("CSV_PATH").unwrap_or_else(|_| {
        env::var("HOME")
            .map(|h| format!("{}/code/freellmkeys.csv", h))
            .unwrap_or_else(|_| "llmkeys.csv".to_string())
    });
    let bind_addr = env::var("BASE_URL")
        .unwrap_or_else(|_| "http://0.0.0.0:3001/v1".to_string());
    let expected_api_key = env::var("API_KEY").unwrap_or_default();
    let cache_enabled = std::env::args().any(|a| a == "--cache");
    if cache_enabled {
        info!("Prompt caching enabled (--cache): cache_control hints will be injected for Anthropic requests");
    }
    let compress_enabled = std::env::args().any(|a| a == "--compress");
    if compress_enabled {
        info!("Compress mode enabled (--compress): caveman system prompt will be injected into all requests");
    }

    let providers = load_providers(&csv_path);
    if providers.is_empty() {
        panic!("No providers loaded from CSV");
    }

    info!(
        bind_addr = %bind_addr,
        providers = providers.len(),
        "Starting LLM Key Rotator"
    );

    let mut client_builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(std::time::Duration::from_secs(90));

    if env::var("USE_CUSTOM_DNS").is_ok() {
        info!("Using custom DNS resolver (Google DNS + Cloudflare)");
        client_builder = client_builder.dns_resolver(Arc::new(dns::GoogleDnsResolver::new()));
    }

    let provider_count = providers.len();
    let state = AppState {
        providers: Arc::new(providers),
        current_index: Arc::new(AtomicUsize::new(0)),
        client: client_builder.build().expect("Failed to build HTTP client"),
        stats: Arc::new(Stats::new(provider_count)),
        expected_api_key,
        cache_enabled,
        compress_enabled,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/v1/*path", post(proxy_openai))
        .route("/anthropic/*path", post(proxy_anthropic))
        .fallback(catch_all)
        .with_state(state);

    let addr: SocketAddr = bind_addr
        .replace("http://", "")
        .replace("https://", "")
        .split('/')
        .next()
        .unwrap_or("0.0.0.0:3001")
        .parse()
        .expect("Invalid bind address");

    info!(addr = %addr, "Listening");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}
