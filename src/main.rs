//! Krill: Recursive Skill-Augmented RL for LLM Agents
//!
//! Provides subcommands for each phase of the pipeline:
//!
//! - `train`    -- Run the full training pipeline (all 4 phases)
//! - `collect`  -- Phase 1: Collect trajectories with the base model
//! - `distill`  -- Phase 2: Distill skills from collected trajectories
//! - `sft`      -- Phase 3: Cold-start supervised fine-tuning
//! - `rl`       -- Phase 4: RL training with recursive skill evolution
//! - `inspect`  -- Inspect a saved skill bank

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use krill::config::SkillRLConfig;
use krill::env::alfworld::MockAlfWorldEnv;
use krill::env::webshop::MockWebShopEnv;
use krill::env::AnyEnv;
use krill::skill::library::SkillBank;
use krill::training::TrainingPipeline;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Krill: Recursive Skill-Augmented RL for LLM Agents
#[derive(Parser)]
#[command(name = "krill", version, about)]
struct Cli {
    /// Path to a JSON configuration file (uses defaults if not provided).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Which environment to use.
    #[arg(long, global = true, default_value = "alfworld")]
    env: EnvChoice,

    /// Use mock environments instead of connecting to live servers.
    #[arg(long, global = true, default_value_t = true)]
    mock: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum EnvChoice {
    Alfworld,
    Webshop,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the full SkillRL training pipeline (phases 1-4).
    Train,

    /// Phase 1: Collect trajectories with the current policy.
    Collect {
        /// Number of episodes to collect.
        #[arg(long, default_value_t = 64)]
        episodes: usize,

        /// Path to save collected trajectories.
        #[arg(long, default_value = "data/trajectories.json")]
        output: PathBuf,
    },

    /// Phase 2: Distill skills from collected trajectories.
    Distill {
        /// Path to the trajectory file produced by `collect`.
        #[arg(long, default_value = "data/trajectories.json")]
        trajectories: PathBuf,

        /// Path to save the resulting skill bank.
        #[arg(long, default_value = "data/skill_bank.json")]
        output: PathBuf,
    },

    /// Phase 3: Cold-start supervised fine-tuning.
    Sft {
        /// Path to the skill bank (from `distill`).
        #[arg(long, default_value = "data/skill_bank.json")]
        skill_bank: PathBuf,

        /// Number of SFT examples to generate.
        #[arg(long, default_value_t = 7500)]
        num_examples: usize,
    },

    /// Phase 4: RL training with recursive skill evolution.
    Rl {
        /// Path to the skill bank (from `distill` or a previous checkpoint).
        #[arg(long, default_value = "data/skill_bank.json")]
        skill_bank: PathBuf,
    },

    /// Inspect a saved skill bank.
    Inspect {
        /// Path to the skill bank JSON file.
        #[arg(default_value = "data/skill_bank.json")]
        path: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise tracing (reads RUST_LOG env var, defaults to info).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Load or create configuration.
    let mut config = match &cli.config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read config from {}", path.display()))?;
            serde_json::from_str::<SkillRLConfig>(&text)
                .with_context(|| format!("Failed to parse config from {}", path.display()))?
        }
        None => SkillRLConfig::default(),
    };

    // Fill in API keys from environment variables when not set in the config file.
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if config.model.policy_api_key.is_empty() {
            config.model.policy_api_key = key.clone();
        }
        if config.model.teacher_api_key.is_empty() {
            config.model.teacher_api_key = key.clone();
        }
        if config.model.embedding_api_key.is_empty() {
            config.model.embedding_api_key = key;
        }
    }

    match cli.command {
        Commands::Train => cmd_train(&config, &cli.env, cli.mock).await,
        Commands::Collect { episodes, output } => {
            cmd_collect(&config, &cli.env, cli.mock, episodes, &output).await
        }
        Commands::Distill {
            trajectories,
            output,
        } => cmd_distill(&config, &trajectories, &output).await,
        Commands::Sft {
            skill_bank,
            num_examples,
        } => cmd_sft(&config, &skill_bank, num_examples).await,
        Commands::Rl { skill_bank } => cmd_rl(&config, &cli.env, cli.mock, &skill_bank).await,
        Commands::Inspect { path } => cmd_inspect(&path),
    }
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async fn cmd_train(config: &SkillRLConfig, env_choice: &EnvChoice, mock: bool) -> Result<()> {
    tracing::info!("Starting full SkillRL training pipeline");

    let pipeline = TrainingPipeline::new(config.clone());
    let mut env = create_env(env_choice, mock);

    pipeline.run(&mut env).await?;

    tracing::info!("Training pipeline finished");
    Ok(())
}

async fn cmd_collect(
    config: &SkillRLConfig,
    env_choice: &EnvChoice,
    mock: bool,
    episodes: usize,
    output: &PathBuf,
) -> Result<()> {
    tracing::info!(episodes, "Collecting trajectories");

    let pipeline = TrainingPipeline::new(config.clone());
    let mut env = create_env(env_choice, mock);

    let (successful, failed) = pipeline.collect_trajectories(&mut env, episodes).await?;

    // Save both to one file.
    let all: Vec<_> = successful.into_iter().chain(failed).collect();

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&all)?;
    std::fs::write(output, json)?;

    tracing::info!(path = %output.display(), count = all.len(), "Saved trajectories");
    Ok(())
}

async fn cmd_distill(
    config: &SkillRLConfig,
    trajectories_path: &PathBuf,
    output: &PathBuf,
) -> Result<()> {
    tracing::info!("Distilling skills from trajectories");

    let text = std::fs::read_to_string(trajectories_path)
        .with_context(|| format!("Failed to read {}", trajectories_path.display()))?;

    let trajectories: Vec<krill::trajectory::types::Trajectory> =
        serde_json::from_str(&text).context("Failed to parse trajectories")?;

    let successful: Vec<_> = trajectories.iter().filter(|t| t.success).cloned().collect();
    let failed: Vec<_> = trajectories.iter().filter(|t| !t.success).cloned().collect();

    let pipeline = TrainingPipeline::new(config.clone());
    let skill_bank = pipeline.distill_skills(&successful, &failed).await?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    skill_bank.save_to_file(output)?;

    tracing::info!(
        skills = skill_bank.len(),
        path = %output.display(),
        "Skill bank saved"
    );
    Ok(())
}

async fn cmd_sft(
    config: &SkillRLConfig,
    skill_bank_path: &PathBuf,
    _num_examples: usize,
) -> Result<()> {
    tracing::info!("Running cold-start SFT");

    let skill_bank = SkillBank::load_from_file(skill_bank_path)?;

    let pipeline = TrainingPipeline::new(config.clone());
    let tasks: Vec<String> = skill_bank
        .task_categories()
        .into_iter()
        .map(|cat| format!("Representative task for {cat}"))
        .collect();

    pipeline.cold_start_sft(&skill_bank, &tasks).await?;

    tracing::info!("SFT phase complete");
    Ok(())
}

async fn cmd_rl(
    config: &SkillRLConfig,
    env_choice: &EnvChoice,
    mock: bool,
    skill_bank_path: &PathBuf,
) -> Result<()> {
    tracing::info!("Running RL training with recursive skill evolution");

    let mut skill_bank = SkillBank::load_from_file(skill_bank_path)?;
    let pipeline = TrainingPipeline::new(config.clone());
    let mut env = create_env(env_choice, mock);

    let metrics = pipeline.rl_training(&mut env, &mut skill_bank).await?;

    if let Some(last) = metrics.last() {
        tracing::info!(
            final_success_rate = format!("{:.2}%", last.success_rate * 100.0),
            total_epochs = metrics.len(),
            skills = last.skill_bank_size,
            "RL training complete"
        );
    }

    Ok(())
}

fn cmd_inspect(path: &PathBuf) -> Result<()> {
    let skill_bank = SkillBank::load_from_file(path)?;

    println!("Skill Bank: {}", path.display());
    println!("  Total skills: {}", skill_bank.len());
    println!(
        "  General skills: {}",
        skill_bank.get_general_skills().len()
    );
    println!("  Evolution cycle: {}", skill_bank.current_cycle());
    println!();

    let counts = skill_bank.skill_count_by_category();
    println!("Skills by category:");
    for (cat, count) in &counts {
        println!("  {cat}: {count}");
    }
    println!();

    println!("General skills:");
    for skill in skill_bank.get_general_skills() {
        println!("  [{id}] {name}", id = &skill.id[..8], name = skill.name);
        println!("    Principle: {}", skill.principle);
        println!("    Apply when: {}", skill.when_to_apply);
        println!();
    }

    for cat in skill_bank.task_categories() {
        let skills = skill_bank.get_task_skills(&cat);
        if !skills.is_empty() {
            println!("Task-specific skills ({cat}):");
            for skill in skills {
                println!("  [{id}] {name}", id = &skill.id[..8], name = skill.name);
                println!("    Principle: {}", skill.principle);
                println!("    Apply when: {}", skill.when_to_apply);
                println!();
            }
        }
    }

    let history = skill_bank.history();
    if !history.is_empty() {
        println!("Evolution history ({} entries):", history.len());
        for entry in history.iter().take(10) {
            println!(
                "  Cycle {}: skill {} at {}",
                entry.evolution_cycle,
                &entry.skill_id[..8],
                entry.added_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
        }
        if history.len() > 10 {
            println!("  ... and {} more", history.len() - 10);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Environment construction
// ---------------------------------------------------------------------------

fn create_env(choice: &EnvChoice, mock: bool) -> AnyEnv {
    match (choice, mock) {
        (EnvChoice::Alfworld, true) => {
            tracing::info!("Using mock ALFWorld environment");
            AnyEnv::MockAlfWorld(MockAlfWorldEnv::new())
        }
        (EnvChoice::Webshop, true) => {
            tracing::info!("Using mock WebShop environment");
            AnyEnv::MockWebShop(MockWebShopEnv::new())
        }
        (EnvChoice::Alfworld, false) => {
            tracing::info!("Using live ALFWorld environment");
            AnyEnv::AlfWorld(krill::env::alfworld::AlfWorldEnv::new(
                "http://localhost:3000",
                50,
            ))
        }
        (EnvChoice::Webshop, false) => {
            tracing::info!("Using live WebShop environment");
            AnyEnv::WebShop(krill::env::webshop::WebShopEnv::new(
                "http://localhost:3001",
                30,
            ))
        }
    }
}
