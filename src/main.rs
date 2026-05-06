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

fn rotate_provider(state: &AppState) {
    let idx = state.current_index.fetch_add(1, Ordering::Relaxed);
    let provider_name = &state.providers[idx % state.providers.len()].name;
    warn!(from = provider_name, "Rotating to next provider");
    state
        .stats
        .blocking_lock()
        .total_rotations
        .fetch_add(1, Ordering::Relaxed);
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

async fn proxy_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
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

        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));

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
                                let event = axum::response::sse::Event::default()
                                    .data(String::from_utf8_lossy(&bytes).to_string());
                                Ok(event)
                            }
                            Err(e) => {
                                error!(error = %e, "Stream error");
                                Err(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    e.to_string(),
                                ))
                            }
                        });
                        return Sse::new(stream)
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
            rotate_provider(&state);
        }
    }

    error!("All providers exhausted");
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
    let path = path.strip_prefix("/v1").unwrap_or(path);
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
        .route("/chat/completions", post(proxy_chat_completions))
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
