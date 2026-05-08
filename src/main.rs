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
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    env,
    error::Error,
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
    use std::collections::HashSet;

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

                if role == "tool" {
                    messages.push(content.unwrap_or(&Value::Null).clone());
                    continue;
                }

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

                        for tr in &tool_results {
                            messages.push(tr.clone());
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
                    Some("any") => serde_json::json!("any"),
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
                stop_reason = match reason {
                    "stop" => "end_turn",
                    "length" => "max_tokens",
                    "tool_calls" => "tool_use",
                    _ => "end_turn",
                };
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

    pub fn sse_chunk_to_anthropic(openai_chunk: &str) -> Vec<(String, String)> {
        let mut events = Vec::new();

        let chunk: Value = match serde_json::from_str(openai_chunk) {
            Ok(v) => v,
            Err(_) => return events,
        };

        let mut has_text = false;
        let mut opened_tool_indices: HashSet<usize> = HashSet::new();
        let mut stop_reason: Option<String> = None;

        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            if !has_text {
                                events.push((
                                    "content_block_start".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": { "type": "text", "text": "" }
                                    }).to_string(),
                                ));
                                has_text = true;
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

                            if !opened_tool_indices.contains(&tc_index) {
                                events.push((
                                    "content_block_start".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_start",
                                        "index": tc_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tc_id,
                                            "name": tc_name,
                                            "input": {}
                                        }
                                    }).to_string(),
                                ));
                                opened_tool_indices.insert(tc_index);
                            }

                            if !tc_args.is_empty() {
                                events.push((
                                    "content_block_delta".to_string(),
                                    serde_json::json!({
                                        "type": "content_block_delta",
                                        "index": tc_index,
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
            if has_text {
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

#[derive(Debug, Default, Serialize)]
struct Stats {
    total_requests: AtomicUsize,
    total_errors: AtomicUsize,
    total_rotations: AtomicUsize,
    provider_failures: HashMap<String, usize>,
}

#[derive(Clone)]
struct AppState {
    providers: Arc<Vec<Provider>>,
    current_index: Arc<AtomicUsize>,
    client: reqwest::Client,
    stats: Arc<Mutex<Stats>>,
    expected_api_key: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
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

fn get_current_provider(state: &AppState) -> &Provider {
    let idx = state.current_index.load(Ordering::Relaxed);
    &state.providers[idx % state.providers.len()]
}

async fn rotate_provider(state: &AppState) {
    let idx = state.current_index.fetch_add(1, Ordering::Relaxed);
    let new_idx = idx + 1;
    let from_name = &state.providers[idx % state.providers.len()].name;
    let to_name = &state.providers[new_idx % state.providers.len()].name;
    warn!(from = from_name, to = to_name, "Rotating to next provider");
    let stats = state.stats.lock().await;
    stats.total_rotations.fetch_add(1, Ordering::Relaxed);
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
    let provider = get_current_provider(&state);
    let stats = state.stats.lock().await;
    Json(serde_json::json!({
        "status": "ok",
        "providers": state.providers.len(),
        "current_provider": provider.name,
        "total_requests": stats.total_requests.load(Ordering::Relaxed),
        "total_errors": stats.total_errors.load(Ordering::Relaxed),
        "total_rotations": stats.total_rotations.load(Ordering::Relaxed),
    }))
}

async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.stats.lock().await;
    Json(serde_json::json!({
        "total_requests": stats.total_requests.load(Ordering::Relaxed),
        "total_errors": stats.total_errors.load(Ordering::Relaxed),
        "total_rotations": stats.total_rotations.load(Ordering::Relaxed),
        "provider_failures": &stats.provider_failures,
    }))
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

    {
        let stats = state.stats.lock().await;
        stats.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    let streaming = is_streaming_request(&body);
    let max_retries = state.providers.len();

    for attempt in 0..max_retries {
        let provider = get_current_provider(&state);
        body["model"] = Value::String(provider.model.clone());

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

        let request_builder = state
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        match request_builder.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    if streaming {
                        let stream = resp.bytes_stream().map(|result| match result {
                            Ok(bytes) => {
                                let text = String::from_utf8_lossy(&bytes).to_string();
                                let mut events = Vec::new();
                                for line in text.lines() {
                                    let line = line.trim();
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
                            Err(e) => {
                                error!(error = %e, "Stream error");
                                futures::stream::iter(vec![Err(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    e.to_string(),
                                ))])
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
                        "Provider returned error"
                    );
                }
            }
            Err(e) => {
                error!(error = %e, provider = %provider.name, "Request failed");
            }
        }

        {
            let mut stats = state.stats.lock().await;
            stats.total_errors.fetch_add(1, Ordering::Relaxed);
            *stats
                .provider_failures
                .entry(provider.name.clone())
                .or_insert(0) += 1;
        }

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
    _uri: Uri,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(resp) = auth_check(&state, &headers) {
        return resp;
    }

    {
        let stats = state.stats.lock().await;
        stats.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    let streaming = is_streaming_request(&body);
    let max_retries = state.providers.len();
    let openai_body = anthropic::request_to_openai(&mut body);

    for attempt in 0..max_retries {
        let provider = get_current_provider(&state);
        let mut forward_body = openai_body.clone();
        forward_body["model"] = Value::String(provider.model.clone());

        info!(
            attempt = attempt + 1,
            provider = %provider.name,
            model = %provider.model,
            streaming = streaming,
            "Forwarding Anthropic request"
        );

        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));

        let request_builder = state
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .json(&forward_body);

        match request_builder.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    if streaming {
                        use std::sync::atomic::AtomicBool;
                        let msg_id = Arc::new(std::sync::Mutex::new(String::new()));
                        let msg_model = Arc::new(std::sync::Mutex::new(String::new()));
                        let sent_start = Arc::new(AtomicBool::new(false));

                        let msg_id_clone = msg_id.clone();
                        let msg_model_clone = msg_model.clone();
                        let sent_start_clone = sent_start.clone();

                        let mut buf = String::new();
                        let stream = resp.bytes_stream().map(move |result| {
                            match result {
                                Ok(bytes) => {
                                    buf.push_str(&String::from_utf8_lossy(&bytes));
                                    let mut events = Vec::new();
                                    while let Some(pos) = buf.find('\n') {
                                        let line = buf[..pos].trim().to_string();
                                        buf = buf[pos + 1..].to_string();
                                        if line.starts_with("data: ") {
                                            let data = &line[6..];
                                            if data == "[DONE]" {
                                                continue;
                                            }
                                            if !sent_start_clone.swap(true, Ordering::SeqCst) {
                                                if let Ok(v) = serde_json::from_str::<Value>(data) {
                                                    let mut id = msg_id_clone.lock().unwrap();
                                                    let mut model = msg_model_clone.lock().unwrap();
                                                    *id = v.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                    *model = v.get("model").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                }
                                                let id = msg_id_clone.lock().unwrap().clone();
                                                let model = msg_model_clone.lock().unwrap().clone();
                                                let (et, ed) = anthropic::sse_message_start(&id, &model);
                                                events.push(Ok(
                                                    axum::response::sse::Event::default()
                                                        .event(et)
                                                        .data(ed),
                                                ));
                                            }
                                            for (event_type, event_data) in anthropic::sse_chunk_to_anthropic(data) {
                                                events.push(Ok(
                                                    axum::response::sse::Event::default()
                                                        .event(event_type)
                                                        .data(event_data),
                                                ));
                                            }
                                        }
                                    }
                                    futures::stream::iter(events)
                                }
                                Err(e) => {
                                    error!(error = %e, "Anthropic stream error");
                                    buf.clear();
                                    futures::stream::iter(vec![Err(std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        e.to_string(),
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
                            Ok(json) => {
                                let anthropic_resp = anthropic::response_to_anthropic(&json);
                                return Json(anthropic_resp).into_response();
                            }
                            Err(e) => {
                                error!(error = %e, provider = %provider.name, "Failed to parse Anthropic response");
                            }
                        }
                    }
                } else {
                    let err_body = resp.text().await.unwrap_or_default();
                    error!(
                        status = %status,
                        provider = %provider.name,
                        body = %err_body,
                        "Provider returned error for Anthropic request"
                    );
                }
            }
            Err(e) => {
                error!(error = %e, provider = %provider.name, "Anthropic request failed");
            }
        }

        {
            let mut stats = state.stats.lock().await;
            stats.total_errors.fetch_add(1, Ordering::Relaxed);
            *stats
                .provider_failures
                .entry(provider.name.clone())
                .or_insert(0) += 1;
        }

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

    {
        let stats = state.stats.lock().await;
        stats.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    let provider = get_current_provider(&state);
    let path = uri.path();
    let path = path
        .strip_prefix("/v1")
        .or_else(|| path.strip_prefix("/anthropic"))
        .unwrap_or(path);
    let url = format!("{}{}", provider.base_url.trim_end_matches('/'), path);

    info!(provider = %provider.name, path = path, url = %url, "Proxying request");

    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "Failed to read body");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to read body".to_string(),
                }),
            )
                .into_response();
        }
    };

    let body_bytes = if let Ok(mut json) = serde_json::from_slice::<Value>(&body_bytes) {
        json["model"] = Value::String(provider.model.clone());
        serde_json::to_vec(&json).unwrap_or(body_bytes.to_vec())
    } else {
        body_bytes.to_vec()
    };

    match state
        .client
        .request(method, &url)
        .header("Authorization", format!("Bearer {}", provider.api_key))
        .header(
            "Content-Type",
            headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json"),
        )
        .body(body_bytes)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            match resp.bytes().await {
                Ok(bytes) => {
                    let mut builder = Response::builder().status(status);
                    for (key, value) in resp_headers.iter() {
                        if key != "transfer-encoding" && key != "content-length" {
                            builder = builder.header(key, value);
                        }
                    }
                    builder.body(Body::from(bytes)).unwrap_or_else(|e| {
                        error!(error = %e, "Failed to build response");
                        StatusCode::INTERNAL_SERVER_ERROR.into_response()
                    })
                }
                Err(e) => {
                    error!(error = %e, "Failed to read response body");
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(ErrorResponse {
                            error: "Upstream error".to_string(),
                        }),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => {
            let mut err_msg = format!("{}", e);
            let mut src = e.source();
            while let Some(s) = src {
                err_msg.push_str(&format!(" | source: {}", s));
                src = s.source();
            }
            error!(error = %err_msg, provider = %provider.name, url = %url, "Proxy request failed");
            {
                let stats = state.stats.lock().await;
                stats.total_errors.fetch_add(1, Ordering::Relaxed);
            }
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    error: err_msg,
                }),
            )
                .into_response()
        }
    }
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

    let providers = load_providers(&csv_path);
    if providers.is_empty() {
        panic!("No providers loaded from CSV");
    }

    info!(
        bind_addr = %bind_addr,
        providers = providers.len(),
        "Starting LLM Key Rotator"
    );

    let state = AppState {
        providers: Arc::new(providers),
        current_index: Arc::new(AtomicUsize::new(0)),
        client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .dns_resolver(Arc::new(dns::GoogleDnsResolver::new()))
            .build()
            .expect("Failed to build HTTP client"),
        stats: Arc::new(Mutex::new(Stats::default())),
        expected_api_key,
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
