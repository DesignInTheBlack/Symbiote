use super::super::*;

impl Kernel {
    pub(crate) fn tool_gate_decision(
        &self,
        tool_name: &str,
        args_json: &str,
        settings: &crate::models::Settings,
        validate_args: bool,
    ) -> ToolGateDecision {
        if !self.is_known_tool_name(tool_name) {
            return ToolGateDecision::block(
                "UNKNOWN_TOOL",
                None,
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if !self.is_allowed_tool_name(tool_name, settings) {
            return ToolGateDecision::block(
                "TOOL_DISABLED",
                None,
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if validate_args {
            if let Err(reason) = validate_tool_call_args(tool_name, args_json) {
                return ToolGateDecision::block(
                    "TOOL_ARGS_INVALID",
                    Some(reason),
                    Some(TOOL_FAILURE_KIND_PLANNING),
                );
            }
        }
        ToolGateDecision::allow()
    }
}
