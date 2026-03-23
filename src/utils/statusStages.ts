import statusStages from "../../shared/status_stages.json";

export type StatusStageId = typeof statusStages.stages[number]["id"];

export type StatusStage = {
  id: StatusStageId;
  label: string;
  default_detail?: string;
};

export const STATUS_STAGES: StatusStage[] = statusStages.stages as StatusStage[];

const STAGE_MAP = new Map<string, StatusStage>(STATUS_STAGES.map((stage) => [stage.id, stage]));

export const getStatusStage = (id?: string | null) => {
  if (!id) return undefined;
  return STAGE_MAP.get(id);
};

export const formatStatusDetail = (id?: string | null, detail?: string | null) => {
  const stage = getStatusStage(id);
  if (!stage) {
    return detail?.trim() || "Working";
  }
  if (detail && detail.trim().length > 0) {
    if (detail.trim() === "user_abort") {
      return id === "cancelled" ? "Cancelled by user" : (stage.default_detail || stage.label);
    }
    return detail.trim();
  }
  return stage.default_detail || stage.label;
};

export const formatStatusLabel = (id?: string | null) => {
  const stage = getStatusStage(id);
  return stage?.label || "Working";
};
