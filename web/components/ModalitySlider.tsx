"use client";

import { useState } from "react";
import { useCopy } from "@/components/Language";

export function ModalitySlider({ sample }: { sample: string | null }) {
  const { t } = useCopy();
  const [position, setPosition] = useState(50);
  if (!sample) {
    return <p className="awaiting">{t.noPair}</p>;
  }
  return (
    <div className="compare">
      <div className="slider">
        <div className="pane rgb" style={{ backgroundImage: `url(${sample})` }} />
        <div className="pane ir" style={{ backgroundImage: `url(${sample})`, clipPath: `inset(0 0 0 ${position}%)` }} />
        <div className="divider" style={{ left: `${position}%` }} aria-hidden="true" />
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={position}
        aria-label={t.sliderLabel}
        onChange={(event) => setPosition(Number(event.target.value))}
      />
      <div className="compare-labels">
        <span>{t.visible}</span>
        <span>{t.realInfrared}</span>
      </div>
    </div>
  );
}
