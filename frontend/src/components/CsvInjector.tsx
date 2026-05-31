"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Btn, Label, Panel, SectionTitle } from "@/components/ui";

/**
 * CSV file injector for the live stream.
 *
 * Drop a stream_transactions-compatible CSV here (or click to choose) and
 * the backend will queue every row through the same path the Scenario
 * Studio uses — events drain ahead of the D0 stream so they surface within
 * one generator tick. Auto-starts the generator if it's not already running.
 *
 * Designed for the verification CSV produced by ``scripts/verify_detection.py
 * --write-csv data/verification_probes.csv`` but works with any CSV that
 * matches the spec's stream schema.
 */
export default function CsvInjector({
  onInjected,
}: {
  onInjected?: () => void;   // called after a successful injection
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{
    filename: string;
    queued_events: number;
    rows_with_errors: number;
    errors: string[];
    queue_depth: number;
    generator_auto_started: boolean;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (file: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.generatorInjectCsv(file, /* autoStart */ true);
      setResult(r);
      if (onInjected) onInjected();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const onChooseClick = () => inputRef.current?.click();

  const onFileSelected = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) submit(f);
    // Reset so re-selecting the same file fires onChange.
    if (inputRef.current) inputRef.current.value = "";
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) submit(f);
  };

  return (
    <Panel className="mb-4">
      <div className="flex justify-between items-start mb-3">
        <div>
          <Label>Inject Transactions</Label>
          <SectionTitle>Upload a transaction CSV</SectionTitle>
        </div>
        <a
          href={api.generatorSampleProbeCsvUrl()}
          download
          className="text-xs font-mono text-teal underline hover:brightness-125"
          title="Sample: 213 events covering every L1 rule, L2 graph feature, and System-1 typology"
        >
          download sample CSV ↓
        </a>
      </div>

      <p className="text-textDim text-xs leading-relaxed mb-3">
        Drop a <span className="font-mono text-text">stream_transactions.csv</span>-compatible
        file here to push every row into the live pipeline. Use{" "}
        <span className="font-mono text-text">scripts/verify_detection.py --write-csv data/verification_probes.csv</span>{" "}
        to generate a probe stream that exercises every L1 rule and L2 graph feature. The
        generator auto-starts if it isn't already running.
      </p>

      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={onChooseClick}
        className={`rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition ${
          dragOver
            ? "border-teal bg-teal/10"
            : busy
              ? "border-textDim bg-panelHi opacity-60 cursor-wait"
              : "border-border bg-panelHi hover:border-teal/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv,application/vnd.ms-excel"
          className="hidden"
          onChange={onFileSelected}
        />
        {busy ? (
          <div className="text-sm text-textDim">Parsing &amp; queueing …</div>
        ) : (
          <>
            <div className="text-3xl mb-2">📂</div>
            <div className="text-sm font-display font-semibold text-text">
              Drop a CSV here, or click to choose
            </div>
            <div className="text-[11px] font-mono text-textDim mt-1">
              .csv · same schema as stream_transactions.csv
            </div>
          </>
        )}
      </div>

      {/* Result panel */}
      {result && (
        <div className="mt-3 rounded-lg border border-teal/30 bg-teal/10 p-3 text-xs">
          <div className="flex justify-between items-baseline mb-1">
            <span className="font-mono text-teal font-bold">{result.filename}</span>
            <span className="text-textDim">queue depth: {result.queue_depth}</span>
          </div>
          <div className="text-text">
            <span className="font-bold text-teal">{result.queued_events}</span> events queued
            {result.rows_with_errors > 0 && (
              <span className="text-amber-400 ml-2">
                · {result.rows_with_errors} row(s) skipped
              </span>
            )}
            {result.generator_auto_started && (
              <span className="text-amber-300 ml-2">· generator was auto-started</span>
            )}
          </div>
          {result.errors.length > 0 && (
            <details className="mt-2">
              <summary className="text-textDim cursor-pointer">view first {result.errors.length} parse error(s)</summary>
              <ul className="mt-1 ml-3 font-mono text-[10px] text-amber-300/80 space-y-0.5">
                {result.errors.map((e, i) => <li key={i}>· {e}</li>)}
              </ul>
            </details>
          )}
        </div>
      )}

      {/* Error panel */}
      {error && (
        <div className="mt-3 rounded-lg border border-high/30 bg-high/10 p-3 text-xs">
          <div className="font-mono text-high font-bold mb-1">Upload failed</div>
          <div className="text-textDim font-mono break-all">{error}</div>
        </div>
      )}
    </Panel>
  );
}
