export const resumeAudioContext = async (ctx: AudioContext | null) => {
  if (!ctx) return;
  const state = ctx.state as string;
  if (state === "suspended" || state === "interrupted") {
    try {
      await ctx.resume();
    } catch (e) {
      console.warn("[Audio] Failed to resume AudioContext", e);
    }
  }
};
