export const cleanForTTS = (text: string): string => {
  // 1. Remove <silent> blocks
  let clean = text.replace(/<silent>[\s\S]*?<\/silent>/g, "");

  // 1.5 Remove <INTERNAL> blocks (may be unterminated)
  clean = clean.replace(/<INTERNAL>[\s\S]*?(<\/INTERNAL>|$)/g, "");

  // 2. Remove memory and reminder syntax blocks
  clean = clean.replace(/```memory[\s\S]*?```/gi, " ");
  clean = clean.replace(/<memory>[\s\S]*?<\/memory>/gi, " ");
  clean = clean.replace(/<MEMORY_CONTEXT>[\s\S]*?<\/MEMORY_CONTEXT>/g, " ");
  clean = clean.replace(/<EPISODIC_CONTEXT>[\s\S]*?<\/EPISODIC_CONTEXT>/g, " ");
  clean = clean.replace(/\[\[(REMINDER:CREATED|TASK:CREATE)[\s\S]*?\]\]/g, " ");
  clean = clean.replace(/<<\s*(MEMORY|CLARIFY|RESOLVE)\s*>>/gi, " ");
  clean = clean.replace(/<attribution>[\s\S]*?<\/attribution>/gi, " ");
  clean = clean.replace(/<state_ref>[\s\S]*?<\/state_ref>/gi, " ");

  // 2. Remove <code> tags (keeping placeholder)
  clean = clean.replace(/<code>[\s\S]*?<\/code>/g, " I've output code in a code block for you. ");

  // 3. Remove Markdown Code Blocks (```...```)
  clean = clean.replace(/```[\s\S]*?```/g, " I've output code in a code block for you. ");

  // 4. Remove Inline Markdown (`...`)
  clean = clean.replace(/`[^`]*?`/g, "");

  // 5. Cleanup
  clean = clean.replace(/<\/?silent>/g, "").replace(/<\/?code>/g, "").replace(/```/g, "").replace(/`/g, "");
  return clean;
};

export const hasUnclosedTags = (text: string): boolean => {
  const openCode = (text.match(/<code>/g) || []).length;
  const closeCode = (text.match(/<\/code>/g) || []).length;
  if (openCode > closeCode) return true;

  const openSilent = (text.match(/<silent>/g) || []).length;
  const closeSilent = (text.match(/<\/silent>/g) || []).length;
  if (openSilent > closeSilent) return true;

  const openMemory = (text.match(/<memory>/gi) || []).length;
  const closeMemory = (text.match(/<\/memory>/gi) || []).length;
  if (openMemory > closeMemory) return true;

  const openMemoryContext = (text.match(/<MEMORY_CONTEXT>/g) || []).length;
  const closeMemoryContext = (text.match(/<\/MEMORY_CONTEXT>/g) || []).length;
  if (openMemoryContext > closeMemoryContext) return true;

  const openEpisodicContext = (text.match(/<EPISODIC_CONTEXT>/g) || []).length;
  const closeEpisodicContext = (text.match(/<\/EPISODIC_CONTEXT>/g) || []).length;
  if (openEpisodicContext > closeEpisodicContext) return true;

  // Markdown Backticks: Count occurrences. Odd = Open.
  const backticks = (text.match(/```/g) || []).length;
  if (backticks % 2 !== 0) return true;

  return false;
};
