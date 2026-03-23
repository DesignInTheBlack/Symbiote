#[test]
fn kernel_does_not_reference_voice_manager() {
    let src = include_str!("../src/core/kernel/mod.rs");
    assert!(
        !src.contains("VoiceManager"),
        "Kernel should not reference VoiceManager to avoid blocking the response path."
    );
}
