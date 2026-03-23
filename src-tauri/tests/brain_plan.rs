use symbiote_lib::core::kernel::sanitize_user_output;

#[test]
fn sanitize_strips_diagnostics_and_workspace_scaffold() {
    let input = r#"
Tool Manifest:
- get_system_logs
focus: testing
open_questions: why
User: hello
Assistant: hi
<INTERNAL>secret</INTERNAL>
"#;
    let (cleaned, changed, _) = sanitize_user_output(input, false, Some("Nova"), Some("Ken"));
    assert!(changed, "Expected sanitizer to modify output");
    let lower = cleaned.to_lowercase();
    assert!(!lower.contains("tool manifest"), "Diagnostics should be removed");
    assert!(!lower.contains("focus:"), "Workspace scaffold should be removed");
    assert!(!lower.contains("user:"), "Role labels should be removed");
    assert!(!lower.contains("assistant:"), "Role labels should be removed");
    assert!(!lower.contains("secret"), "Internal blocks should be removed");
}

#[test]
fn sanitize_allows_system_overview_when_requested() {
    let input = "Symbiote System Overview:\n- Kernel: orchestrates runs.";
    let (cleaned, changed, _) = sanitize_user_output(input, true, Some("Nova"), Some("Ken"));
    assert!(!changed, "Diagnostics should remain when explicitly allowed");
    assert!(cleaned.contains("Symbiote System Overview"));
}

#[test]
fn sanitize_strips_custom_role_labels() {
    let input = "Nova: Hello\nKen: Hi there\n";
    let (cleaned, changed, _) = sanitize_user_output(input, false, Some("Nova"), Some("Ken"));
    assert!(changed, "Expected role labels to be removed");
    assert!(cleaned.trim().is_empty(), "Expected all role-labeled lines to be stripped");
}
