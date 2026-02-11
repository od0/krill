//! Embedding model client for computing vector representations of text.
//!
//! Uses the OpenAI-compatible `/embeddings` endpoint to produce dense vectors
//! that are used for cosine-similarity skill retrieval:
//!
//!   `S_ret = TopK({s in S_k : sim(e_d, e_s) > delta}, K)`

use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// A single embedding object returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingObject {
    /// Index within the request batch.
    index: usize,
    /// The embedding vector.
    embedding: Vec<f64>,
}

/// Token usage for an embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Top-level response from the embeddings API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingObject>,
    usage: EmbeddingUsage,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// HTTP client for an OpenAI-compatible embeddings API.
///
/// Wraps [`reqwest::Client`] with the base URL, API key, and model identifier
/// needed to call `POST {base_url}/embeddings`.
#[derive(Debug, Clone)]
pub struct EmbeddingClient {
    http_client: reqwest::Client,
    base_url: String,
    api_key: String,
    model_id: String,
}

impl EmbeddingClient {
    /// Create a new client.
    ///
    /// # Arguments
    ///
    /// * `base_url`  -- API base URL (e.g. `"https://api.openai.com/v1"`).
    /// * `api_key`   -- Bearer token for authentication.
    /// * `model_id`  -- Embedding model identifier (e.g. `"text-embedding-3-small"`).
    pub fn new(base_url: &str, api_key: &str, model_id: &str) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build reqwest client for embedding");

        Self {
            http_client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model_id: model_id.to_string(),
        }
    }

    /// Get the embedding vector for a single text string.
    pub async fn embed(&self, text: &str) -> Result<Vec<f64>> {
        let url = format!("{}/embeddings", self.base_url);
        debug!(model = %self.model_id, text_len = text.len(), "embedding single text");

        let body = serde_json::json!({
            "model": self.model_id,
            "input": text,
        });

        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to send embedding request")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("embedding API returned {status}: {text}");
        }

        let emb_resp: EmbeddingResponse = resp
            .json()
            .await
            .context("failed to parse embedding response")?;

        let embedding = emb_resp
            .data
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .unwrap_or_default();

        info!(
            model = %self.model_id,
            dim = embedding.len(),
            "embedding computed"
        );

        Ok(embedding)
    }

    /// Get embedding vectors for a batch of texts.
    ///
    /// The API is called once with all texts as an array input.  Results are
    /// returned in the same order as the input slice.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        debug!(
            model = %self.model_id,
            batch_size = texts.len(),
            "embedding batch"
        );

        let body = serde_json::json!({
            "model": self.model_id,
            "input": texts,
        });

        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to send batch embedding request")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("batch embedding API returned {status}: {text}");
        }

        let emb_resp: EmbeddingResponse = resp
            .json()
            .await
            .context("failed to parse batch embedding response")?;

        // The API may return objects out of order; sort by index.
        let mut sorted = emb_resp.data;
        sorted.sort_by_key(|e| e.index);

        let embeddings: Vec<Vec<f64>> = sorted.into_iter().map(|e| e.embedding).collect();

        info!(
            model = %self.model_id,
            batch_size = embeddings.len(),
            dim = embeddings.first().map(|v| v.len()).unwrap_or(0),
            "batch embedding computed"
        );

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }"#;

        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding.len(), 3);
        assert!((resp.data[0].embedding[0] - 0.1).abs() < 1e-9);
        assert_eq!(resp.usage.prompt_tokens, 10);
    }

    #[test]
    fn test_embedding_response_out_of_order() {
        let json = r#"{
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]}
            ],
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }"#;

        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        let mut sorted = resp.data;
        sorted.sort_by_key(|e| e.index);
        assert!((sorted[0].embedding[0] - 0.1).abs() < 1e-9);
        assert!((sorted[1].embedding[0] - 0.4).abs() < 1e-9);
    }
}
