import { invokeWithTimeout } from "./tauri";

export type ThemeSource = "builtin" | "user";

export interface ThemeOption {
  id: string;
  label: string;
  source: ThemeSource;
  name: string;
}

export const DEFAULT_THEME_ID = "builtin:utopia";
export const THEME_CHANGE_EVENT = "symbiote-theme-change";
const THEME_STYLE_ID = "symbiote-theme";

const BUILTIN_THEMES: ThemeOption[] = [
  { id: "builtin:default", label: "Default", source: "builtin", name: "default" },
  { id: "builtin:ideate", label: "Ideate", source: "builtin", name: "ideate" },
  { id: "builtin:aurora", label: "Aurora", source: "builtin", name: "aurora" },
  { id: "builtin:utopia", label: "Utopia", source: "builtin", name: "utopia" },
  { id: "builtin:incorporated", label: "Incorporated", source: "builtin", name: "incorporated" },
];

const parseThemeId = (id: string): ThemeOption => {
  const [sourceRaw, name] = id.split(":");
  const source = sourceRaw === "user" ? "user" : "builtin";
  const safeName = name || "default";
  const label = source === "builtin" ? capitalize(safeName) : safeName;
  return { id: `${source}:${safeName}`, label, source, name: safeName };
};

const capitalize = (value: string) =>
  value.length === 0 ? value : value.charAt(0).toUpperCase() + value.slice(1);

const ensureThemeStyleTag = () => {
  let tag = document.getElementById(THEME_STYLE_ID) as HTMLStyleElement | null;
  if (!tag) {
    tag = document.createElement("style");
    tag.id = THEME_STYLE_ID;
    document.head.appendChild(tag);
  }
  return tag;
};

export const listThemeOptions = async (): Promise<{ options: ThemeOption[]; dir: string | null }> => {
  let userThemes: string[] = [];
  let dir: string | null = null;
  try {
    const res = await invokeWithTimeout<{ themes: string[]; dir: string }>("list_themes");
    userThemes = res.themes || [];
    dir = res.dir || null;
  } catch (_e) {
    // Ignore theme listing errors; fall back to built-ins only.
  }

  const userOptions = userThemes.map((name) => ({
    id: `user:${name}`,
    label: name,
    source: "user" as ThemeSource,
    name,
  }));

  return { options: [...BUILTIN_THEMES, ...userOptions], dir };
};

export const getCssVar = (name: string, fallback: string) => {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
};

const loadBuiltinTheme = async (name: string) => {
  const res = await fetch(`/themes/${name}.css`);
  if (!res.ok) {
    throw new Error(`Theme not found: ${name}`);
  }
  return res.text();
};

const loadUserTheme = async (name: string) => {
  return invokeWithTimeout<string>("read_theme_file", { name });
};

export const applyTheme = async (themeId: string | null) => {
  const resolved = themeId || DEFAULT_THEME_ID;
  const parsed = parseThemeId(resolved);
  try {
    const css = parsed.source === "user"
      ? await loadUserTheme(parsed.name)
      : await loadBuiltinTheme(parsed.name);

    const tag = ensureThemeStyleTag();
    tag.textContent = css;
    document.documentElement.dataset.theme = parsed.id;
    window.dispatchEvent(new CustomEvent(THEME_CHANGE_EVENT));
  } catch (_e) {
    if (parsed.id !== DEFAULT_THEME_ID) {
      await applyTheme(DEFAULT_THEME_ID);
    }
  }
};
