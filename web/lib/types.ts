export type Metrics = { ssim: number | null; psnr: number | null; lpips: number | null; ap50: number | null };

export type HistoryRow = { epoch: number; train_loss: number; val_loss: number };

export type FailureRow = {
  sample_id: string;
  category: string;
  mae?: number;
  observation: string | null;
  possible_explanation: string | null;
};

export type Experiment = {
  id: string;
  model: string;
  status: string;
  training_fraction: number;
  config: string;
  metrics: Metrics;
  training: HistoryRow[] | null;
  samples: string[];
  sample: string | null;
  highest_error: FailureRow[];
};

export type Study = {
  project: { title: string; question: string; status: string };
  dataset: { name: string; status: string; counts: Record<string, number> | null; citation: string };
  downstream: { status: string; ap50?: number; precision_at_0_5?: number; recall_at_0_5?: number; ground_truth?: number };
  experiments: Experiment[];
  finding?: string | null;
};
