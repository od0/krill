//! OpenAI-compatible LLM API client.
//!
//! Provides typed request/response structures and methods for chat completion,
//! chat completion with log-probabilities (needed for GRPO importance ratio
//! computation), and standalone log-probability estimation.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// The textual content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Convenience constructor for a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Convenience constructor for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Convenience constructor for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Token-level log-probability information returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token string.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f64,
}

/// Log-probability information attached to a choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceLogProbs {
    /// Per-token log-probability entries.
    pub content: Option<Vec<TokenLogProb>>,
}

/// A single completion choice returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Zero-based index of this choice within the response.
    pub index: usize,
    /// The generated message.
    pub message: ChatMessage,
    /// The reason the model stopped generating (e.g. `"stop"`, `"length"`).
    pub finish_reason: Option<String>,
    /// Optional log-probability information (present when requested).
    #[serde(default)]
    pub logprobs: Option<ChoiceLogProbs>,
}

/// Token usage statistics for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: usize,
    /// Tokens generated in the completion.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
}

/// A chat completion response from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Unique identifier for this completion.
    pub id: String,
    /// The list of generated choices.
    pub choices: Vec<Choice>,
    /// Token usage statistics.
    pub usage: Usage,
}

/// A chat completion response together with aggregated log-probability data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseWithLogProbs {
    /// The underlying chat response.
    pub response: ChatResponse,
    /// Per-token log probabilities for the first choice.
    pub token_log_probs: Vec<f64>,
    /// Sum of all token log probabilities (log P(completion | prompt)).
    pub total_log_prob: f64,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// HTTP client for an OpenAI-compatible chat completions API.
///
/// Wraps [`reqwest::Client`] with the base URL and API key needed to call
/// endpoints such as `/chat/completions`.
///
/// Field names (`api_base`, `api_key`, `http`) are kept public for backward
/// compatibility with existing code that accesses them directly.
#[derive(Debug, Clone)]
pub struct LlmClient {
    /// The base URL for API requests (e.g. `"https://api.openai.com/v1"`).
    pub api_base: String,
    /// The API key used for bearer authentication.
    pub api_key: String,
    /// The underlying HTTP client.
    pub http: reqwest::Client,
}

impl LlmClient {
    /// Create a new client pointing at `base_url` (e.g. `"https://api.openai.com/v1"`).
    pub fn new(base_url: &str, api_key: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build reqwest client");

        Self {
            api_base: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            http,
        }
    }

    // ------------------------------------------------------------------
    // Chat completions
    // ------------------------------------------------------------------

    /// Send a chat completion request and return the parsed response.
    ///
    /// Calls `POST {base_url}/chat/completions`.
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: &[ChatMessage],
        temperature: f64,
        max_tokens: usize,
    ) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.api_base);
        debug!(model, temperature, max_tokens, "sending chat completion request");

        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to send chat completion request")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("chat completion API returned {status}: {text}");
        }

        let chat_response: ChatResponse = resp
            .json()
            .await
            .context("failed to parse chat completion response")?;

        info!(
            model,
            prompt_tokens = chat_response.usage.prompt_tokens,
            completion_tokens = chat_response.usage.completion_tokens,
            "chat completion succeeded"
        );

        Ok(chat_response)
    }

    /// Send a chat completion request with log-probabilities enabled.
    ///
    /// The returned [`ChatResponseWithLogProbs`] includes per-token log
    /// probabilities and a `total_log_prob` field which is the sum over all
    /// generated tokens (i.e. log P(completion | prompt) under the model).
    /// This is needed to compute the importance sampling ratio rho_i in GRPO.
    pub async fn chat_completion_with_logprobs(
        &self,
        model: &str,
        messages: &[ChatMessage],
        temperature: f64,
        max_tokens: usize,
    ) -> Result<ChatResponseWithLogProbs> {
        let url = format!("{}/chat/completions", self.api_base);
        debug!(model, temperature, max_tokens, "sending chat completion request with logprobs");

        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": true,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to send chat completion request with logprobs")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("chat completion (logprobs) API returned {status}: {text}");
        }

        let chat_response: ChatResponse = resp
            .json()
            .await
            .context("failed to parse chat completion (logprobs) response")?;

        // Extract per-token log probabilities from the first choice.
        let token_log_probs = extract_token_log_probs(&chat_response);
        let total_log_prob: f64 = token_log_probs.iter().sum();

        info!(
            model,
            num_tokens = token_log_probs.len(),
            total_log_prob,
            "chat completion with logprobs succeeded"
        );

        Ok(ChatResponseWithLogProbs {
            response: chat_response,
            token_log_probs,
            total_log_prob,
        })
    }

    // ------------------------------------------------------------------
    // Log-probability estimation
    // ------------------------------------------------------------------

    /// Compute the log probability of `completion` given `prompt`.
    ///
    /// This constructs a two-message conversation where the prompt is the user
    /// message and the completion is pre-filled as the assistant reply, then
    /// requests log-probabilities for the completion tokens.  The result is
    /// `log P(completion | prompt)` under the specified model, which is used
    /// to compute the policy ratio rho_i = pi_theta / pi_ref in GRPO.
    ///
    /// **Note:** Not all API providers support the `echo` / logprob-of-existing
    /// approach.  This implementation uses a prompt-then-completion message pair
    /// with `logprobs: true` and `max_tokens: 1` to avoid generating new
    /// tokens, relying on the provider returning log-probs for the supplied
    /// assistant message.  For providers that do not support this pattern,
    /// consider using the completions endpoint or a custom scoring endpoint.
    pub async fn compute_log_prob(
        &self,
        model: &str,
        prompt: &str,
        completion: &str,
    ) -> Result<f64> {
        let url = format!("{}/chat/completions", self.api_base);
        debug!(model, "computing log probability of completion");

        // Build messages: the user prompt followed by the completion as an
        // assistant message.  We ask the model to continue from this point
        // with max_tokens=0 (or 1 for providers that require at least 1) and
        // enable logprobs so we can read the probabilities of the provided
        // assistant tokens.
        let messages = vec![
            ChatMessage::user(prompt),
            ChatMessage::assistant(completion),
        ];

        // First attempt: use the echo-style request that returns logprobs for
        // existing tokens.  We also set temperature to 0 to make the scoring
        // deterministic.
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": true,
            "echo": true,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to send log-prob request")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("log-prob API returned {status}: {text}");
        }

        let chat_response: ChatResponse = resp
            .json()
            .await
            .context("failed to parse log-prob response")?;

        let token_log_probs = extract_token_log_probs(&chat_response);

        if token_log_probs.is_empty() {
            // Fallback: if the provider did not return per-token logprobs,
            // return 0.0 (log(1)) and log a warning.
            tracing::warn!(
                model,
                "API did not return token log-probs; returning 0.0 as fallback"
            );
            return Ok(0.0);
        }

        let total: f64 = token_log_probs.iter().sum();
        debug!(model, total_log_prob = total, "computed log probability");
        Ok(total)
    }

    // ------------------------------------------------------------------
    // Simple text generation (kept for backward compatibility with the
    // existing `model::LlmClient::generate` interface used by the skill
    // module).
    // ------------------------------------------------------------------

    /// Send a user prompt to the model and return the generated text.
    ///
    /// This is a convenience wrapper around [`chat_completion`] that returns
    /// only the text content of the first choice.
    pub async fn generate(&self, prompt: &str, model_id: &str) -> Result<String> {
        let messages = vec![ChatMessage::user(prompt)];
        let resp = self.chat_completion(model_id, &messages, 0.7, 4096).await?;
        let content = resp
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        Ok(content)
    }

    /// Send a user prompt with a system message and return the generated text.
    pub async fn generate_with_system(
        &self,
        prompt: &str,
        system: &str,
        model_id: &str,
    ) -> Result<String> {
        let messages = vec![ChatMessage::system(system), ChatMessage::user(prompt)];
        let resp = self.chat_completion(model_id, &messages, 0.7, 4096).await?;
        let content = resp
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        Ok(content)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract per-token log probabilities from the first choice of a response.
fn extract_token_log_probs(response: &ChatResponse) -> Vec<f64> {
    response
        .choices
        .first()
        .and_then(|choice| choice.logprobs.as_ref())
        .and_then(|lp| lp.content.as_ref())
        .map(|tokens| tokens.iter().map(|t| t.logprob).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful.");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "You are helpful.");

        let usr = ChatMessage::user("Hello");
        assert_eq!(usr.role, "user");

        let asst = ChatMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_extract_token_log_probs_empty() {
        let resp = ChatResponse {
            id: "test".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("hi"),
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
        };
        assert!(extract_token_log_probs(&resp).is_empty());
    }

    #[test]
    fn test_extract_token_log_probs_present() {
        let resp = ChatResponse {
            id: "test".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("hello world"),
                finish_reason: Some("stop".into()),
                logprobs: Some(ChoiceLogProbs {
                    content: Some(vec![
                        TokenLogProb {
                            token: "hello".into(),
                            logprob: -0.5,
                        },
                        TokenLogProb {
                            token: " world".into(),
                            logprob: -1.2,
                        },
                    ]),
                }),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            },
        };
        let lps = extract_token_log_probs(&resp);
        assert_eq!(lps.len(), 2);
        assert!((lps[0] - (-0.5)).abs() < 1e-9);
        assert!((lps[1] - (-1.2)).abs() < 1e-9);
    }

    #[test]
    fn test_chat_response_serialization_roundtrip() {
        let resp = ChatResponse {
            id: "chatcmpl-abc".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("test"),
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: ChatResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, resp.id);
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.usage.total_tokens, 15);
    }
}
