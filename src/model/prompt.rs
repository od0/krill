//! Prompt templates for the SkillRL system.
//!
//! Each function constructs a `Vec<ChatMessage>` ready to be sent to an LLM.
//! The prompts encode the core SkillRL workflow:
//!
//! - **Skill-augmented action**: inject skills into the agent context.
//! - **Skill distillation**: extract strategic knowledge from trajectories.
//! - **Initial library generation**: bootstrap the skill bank.
//! - **Cold-start trajectory generation**: produce teacher demonstrations.
//! - **Skill evolution**: grow the skill bank from failure analysis.

use crate::model::api::ChatMessage;
use crate::skill::types::Skill;

// ---------------------------------------------------------------------------
// Skill-augmented action prompt
// ---------------------------------------------------------------------------

/// Build the main agent prompt that includes skills in context.
///
/// This implements the skill-augmented policy:
///
///   `a_t ~ pi_theta(a_t | o_<=t, d, S_g, S_ret)`
///
/// The system prompt instructs the agent to:
/// 1. Read the task description and observation history.
/// 2. Review the provided general and retrieved skills.
/// 3. Use chain-of-thought reasoning to decide on an action.
/// 4. Output the action in a structured `Action: <action>` format.
pub fn skill_augmented_action_prompt(
    task_description: &str,
    observation_history: &str,
    general_skills: &[Skill],
    retrieved_skills: &[Skill],
) -> Vec<ChatMessage> {
    // Format general skills.
    let general_section = if general_skills.is_empty() {
        "  (none)".to_string()
    } else {
        general_skills
            .iter()
            .enumerate()
            .map(|(i, s)| {
                format!(
                    "  {}. [{}] {}\n     Principle: {}\n     Apply when: {}",
                    i + 1,
                    s.id,
                    s.name,
                    s.principle,
                    s.when_to_apply
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Format retrieved (task-specific) skills.
    let retrieved_section = if retrieved_skills.is_empty() {
        "  (none)".to_string()
    } else {
        retrieved_skills
            .iter()
            .enumerate()
            .map(|(i, s)| {
                format!(
                    "  {}. [{}] {}\n     Principle: {}\n     Apply when: {}",
                    i + 1,
                    s.id,
                    s.name,
                    s.principle,
                    s.when_to_apply
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let system = format!(
        r#"You are an intelligent agent solving tasks step by step.

You have access to a library of skills -- distilled strategic knowledge that should guide your reasoning and actions. Always consider the applicable skills before deciding on an action.

## General Skills (always applicable)
{general_section}

## Retrieved Skills (relevant to the current task)
{retrieved_section}

## Instructions
1. Read the task description and the history of observations carefully.
2. Review the skills listed above. Identify which skills are relevant to the current situation.
3. Think step by step (chain-of-thought). Explain your reasoning, referencing specific skills where applicable.
4. Choose the best action and output it on a line by itself in the format:

   Action: <your action here>

Important:
- Always output exactly one Action line at the end.
- Your reasoning should come before the Action line.
- Apply the skills to inform your decision -- do not ignore them."#
    );

    let user = format!(
        "## Task\n{task_description}\n\n## Observation History\n{observation_history}\n\nWhat is your next action?"
    );

    vec![ChatMessage::system(system), ChatMessage::user(user)]
}

// ---------------------------------------------------------------------------
// Skill distillation: success
// ---------------------------------------------------------------------------

/// Build the prompt for the teacher to extract strategic patterns from
/// successful trajectories.
///
/// The teacher analyzes successful trajectories and distills reusable skills
/// with structured JSON output containing `name`, `principle`, and
/// `when_to_apply` fields.
pub fn skill_distillation_success_prompt(
    trajectories_json: &str,
    task_description: &str,
) -> Vec<ChatMessage> {
    let system = r#"You are an expert AI researcher analyzing successful agent trajectories to extract reusable strategic skills.

Your goal is to identify the key strategic patterns that led to success and distill them into concise, actionable skills that can be applied to similar future tasks.

For each skill you identify, provide a JSON object with these fields:
- "name": A short, descriptive name for the skill (5-10 words).
- "principle": The core strategic insight -- what to do and why it works (1-2 sentences).
- "when_to_apply": A clear description of the situations where this skill should be used (1 sentence).

Output a JSON array of skill objects. Focus on patterns that are:
1. **Transferable** -- applicable beyond this specific task.
2. **Actionable** -- concrete enough to guide behavior.
3. **Non-obvious** -- capturing insights that a naive agent would miss."#;

    let user = format!(
        r#"## Task Description
{task_description}

## Successful Trajectories
{trajectories_json}

Analyze these successful trajectories and extract the key strategic skills that led to success. Return your answer as a JSON array of skill objects."#
    );

    vec![ChatMessage::system(system.to_string()), ChatMessage::user(user)]
}

// ---------------------------------------------------------------------------
// Skill distillation: failure
// ---------------------------------------------------------------------------

/// Build the prompt for the teacher to synthesize failure lessons from
/// unsuccessful trajectories.
///
/// The teacher identifies failure points, flawed reasoning, the correct
/// action, and prevention principles.
pub fn skill_distillation_failure_prompt(
    trajectories_json: &str,
    task_description: &str,
) -> Vec<ChatMessage> {
    let system = r#"You are an expert AI researcher analyzing failed agent trajectories to understand what went wrong and extract defensive skills that prevent similar failures in the future.

For each failure pattern you identify, provide a JSON object with these fields:
- "name": A short, descriptive name for the defensive skill (5-10 words).
- "principle": What the agent should do differently -- the corrected strategy (1-2 sentences).
- "when_to_apply": The situation or warning sign that should trigger this skill (1 sentence).
- "failure_point": A brief description of where the original trajectory went wrong.
- "flawed_reasoning": What incorrect assumption or logic the agent used.
- "correct_action": What the agent should have done instead.

Output a JSON array of skill objects. Focus on:
1. **Root causes** -- identify the fundamental mistake, not just symptoms.
2. **Prevention** -- frame the skill as a proactive check, not a reactive fix.
3. **Generalizability** -- the skill should help across similar tasks, not just this one."#;

    let user = format!(
        r#"## Task Description
{task_description}

## Failed Trajectories
{trajectories_json}

Analyze these failed trajectories. For each distinct failure pattern, identify the failure point, the flawed reasoning, the correct action, and distill a prevention skill. Return your answer as a JSON array of skill objects."#
    );

    vec![ChatMessage::system(system.to_string()), ChatMessage::user(user)]
}

// ---------------------------------------------------------------------------
// Initial library generation
// ---------------------------------------------------------------------------

/// Build the prompt to generate the initial skill library (8-12 general
/// skills) from a mix of successful and failed trajectories.
///
/// The initial library should cover: navigation, object manipulation,
/// state tracking, and error recovery.
///
/// Skills must be: Concise, Actionable, Transferable, Failure-aware.
pub fn initial_library_prompt(
    successful_json: &str,
    failed_json: &str,
) -> Vec<ChatMessage> {
    let system = r#"You are an expert AI researcher bootstrapping an initial skill library for a reinforcement-learning agent. You will analyze a set of successful and failed trajectories to create a diverse set of 8-12 general-purpose skills.

Each skill should be a JSON object with:
- "name": A concise, descriptive name (5-10 words).
- "principle": The strategic insight this skill encodes (1-2 sentences).
- "when_to_apply": When the agent should invoke this skill (1 sentence).
- "category": Must be "general" for all skills in the initial library.

Your skills MUST cover these domains:
1. **Navigation** -- strategies for exploring and moving through environments.
2. **Object manipulation** -- how to interact with objects effectively.
3. **State tracking** -- maintaining awareness of what has been done and what remains.
4. **Error recovery** -- detecting and recovering from mistakes.

Requirements for each skill:
- **Concise**: The principle should be 1-2 sentences max.
- **Actionable**: Specific enough that an agent can follow it.
- **Transferable**: Useful across different task types, not just one scenario.
- **Failure-aware**: Incorporate lessons from the failed trajectories.

Output a JSON array of 8-12 skill objects."#;

    let user = format!(
        r#"## Successful Trajectories
{successful_json}

## Failed Trajectories
{failed_json}

Based on these trajectories, generate an initial library of 8-12 general-purpose skills. Return your answer as a JSON array."#
    );

    vec![ChatMessage::system(system.to_string()), ChatMessage::user(user)]
}

// ---------------------------------------------------------------------------
// Cold-start trajectory generation
// ---------------------------------------------------------------------------

/// Build the prompt for the teacher to generate skill-augmented reasoning
/// traces for cold-start supervised fine-tuning.
///
/// The generated trace should demonstrate: skill retrieval, interpretation,
/// and application within the chain-of-thought.
pub fn cold_start_trajectory_prompt(
    task_description: &str,
    skills_json: &str,
) -> Vec<ChatMessage> {
    let system = r#"You are an expert AI agent demonstrating how to solve tasks by applying a library of strategic skills. Your goal is to produce a high-quality reasoning trace that a student model can learn from during supervised fine-tuning.

Your trace must demonstrate three things:
1. **Skill retrieval**: Identify which skills from the library are relevant to the current step.
2. **Skill interpretation**: Explain how the skill's principle applies to the current situation.
3. **Skill application**: Show how the skill guides your choice of action.

Format your output as a JSON array of step objects, each with:
- "observation": What you observe at this step.
- "reasoning": Your chain-of-thought, explicitly referencing skills by name.
- "skills_used": An array of skill names you applied.
- "action": The action you take.

Make your reasoning detailed and pedagogical -- a student model should be able to learn the skill-application pattern from reading your trace."#;

    let user = format!(
        r#"## Task
{task_description}

## Available Skills
{skills_json}

Generate a complete, step-by-step reasoning trace for this task. At each step, demonstrate how you select and apply relevant skills from the library. Return your answer as a JSON array of step objects."#
    );

    vec![ChatMessage::system(system.to_string()), ChatMessage::user(user)]
}

// ---------------------------------------------------------------------------
// Skill evolution
// ---------------------------------------------------------------------------

/// Build the prompt for the teacher to identify failure patterns not
/// addressed by the current skill library and propose new or refined skills.
///
/// This implements the recursive evolution step of SkillRL:
///
///   `SkillBank <- SkillBank ∪ S_new`
pub fn evolution_prompt(
    failed_trajectories_json: &str,
    current_skills_json: &str,
) -> Vec<ChatMessage> {
    let system = r#"You are an expert AI researcher evolving an agent's skill library through failure analysis.

You will be given:
1. A set of recent failed trajectories.
2. The agent's current skill library.

Your job is to:
1. Identify failure patterns that are NOT addressed by any existing skill.
2. For each unaddressed failure pattern, propose a new skill.
3. Identify existing skills that are close but insufficient, and propose refined versions.

For each new or refined skill, output a JSON object with:
- "name": A concise, descriptive name (5-10 words).
- "principle": The strategic insight (1-2 sentences).
- "when_to_apply": When the agent should invoke this skill (1 sentence).
- "category": Either "general" or a task-specific category string.
- "is_refinement": true if this refines an existing skill, false if entirely new.
- "refines_skill": The name of the existing skill being refined (if is_refinement is true, otherwise null).
- "failure_pattern": A brief description of the failure pattern this skill addresses.

Guidelines:
- Do NOT propose skills that duplicate existing ones.
- Each new skill should address a DISTINCT failure pattern.
- Prefer actionable, specific skills over vague advice.
- Limit your proposals to the most impactful 1-5 new skills.

Output a JSON array of skill proposals."#;

    let user = format!(
        r#"## Recent Failed Trajectories
{failed_trajectories_json}

## Current Skill Library
{current_skills_json}

Analyze the failures above. Identify failure patterns not covered by the current skills and propose new or refined skills to address them. Return your answer as a JSON array of skill proposals."#
    );

    vec![ChatMessage::system(system.to_string()), ChatMessage::user(user)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::types::{Skill, SkillCategory};

    fn sample_skills() -> Vec<Skill> {
        vec![
            Skill::new(
                "Verify before submit",
                "Always verify the current state before submitting.",
                "Before submitting a final answer or action.",
                SkillCategory::General,
            ),
            Skill::new(
                "Explore systematically",
                "Explore the environment methodically rather than randomly.",
                "When entering a new area or starting a task.",
                SkillCategory::TaskSpecific("navigation".into()),
            ),
        ]
    }

    #[test]
    fn test_skill_augmented_action_prompt_structure() {
        let skills = sample_skills();
        let messages = skill_augmented_action_prompt(
            "Find the red ball",
            "Step 1: You see a room with a table.",
            &skills[..1],
            &skills[1..],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");

        // System prompt should contain skill names.
        assert!(messages[0].content.contains("Verify before submit"));
        assert!(messages[0].content.contains("Explore systematically"));
        // System prompt should contain the Action format instruction.
        assert!(messages[0].content.contains("Action:"));
        // User prompt should contain the task.
        assert!(messages[1].content.contains("Find the red ball"));
    }

    #[test]
    fn test_skill_augmented_action_prompt_empty_skills() {
        let messages =
            skill_augmented_action_prompt("Do something", "obs", &[], &[]);
        assert!(messages[0].content.contains("(none)"));
    }

    #[test]
    fn test_distillation_success_prompt_structure() {
        let messages =
            skill_distillation_success_prompt("{}", "Pick up the mug");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("successful"));
        assert!(messages[1].content.contains("Pick up the mug"));
    }

    #[test]
    fn test_distillation_failure_prompt_structure() {
        let messages =
            skill_distillation_failure_prompt("{}", "Pick up the mug");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("failure"));
        assert!(messages[1].content.contains("Pick up the mug"));
    }

    #[test]
    fn test_initial_library_prompt_structure() {
        let messages = initial_library_prompt("{}", "{}");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("8-12"));
        assert!(messages[0].content.contains("Navigation"));
        assert!(messages[0].content.contains("Error recovery"));
    }

    #[test]
    fn test_cold_start_trajectory_prompt_structure() {
        let messages =
            cold_start_trajectory_prompt("Clean the table", "[]");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("Skill retrieval"));
        assert!(messages[0].content.contains("Skill interpretation"));
        assert!(messages[0].content.contains("Skill application"));
        assert!(messages[1].content.contains("Clean the table"));
    }

    #[test]
    fn test_evolution_prompt_structure() {
        let messages = evolution_prompt("[]", "[]");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("failure patterns"));
        assert!(messages[0].content.contains("is_refinement"));
    }
}
