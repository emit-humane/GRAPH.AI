"use client";

import { useEffect, useRef, useState } from "react";
import { SSE_URL } from "@/lib/api";

export interface LiveEvent {
  transaction_id: string;
  sender_account: string;
  receiver_account: string;
  amount: number;
  currency: string;
  timestamp: string;
  transaction_type: string;
  rule_score: number;
  supervised_score: number | null;
  anomaly_score: number | null;
  tgn_score: number | null;
  transaction_risk_score: number;
  group_risk_score: number;
  risk_level: "Low" | "Medium" | "High" | "Critical";
  risk_level_group: string;
  triggered_rules: string[];
  triggered_patterns: string[];
  top_shap_features: string[];
  explanation: string;
  score_breakdown: any;
  alert_id: string | null;
  alerted: boolean;
  tgn_mode: string | null;
  sender_country: string;
  receiver_country: string;
}

interface State {
  events: LiveEvent[];
  alerts: LiveEvent[];
  connected: boolean;
}

export function useEventStream(maxBuffer: number = 60) {
  const [state, setState] = useState<State>({ events: [], alerts: [], connected: false });
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(SSE_URL);
    esRef.current = es;
    es.onopen = () => setState(s => ({ ...s, connected: true }));
    es.onerror = () => setState(s => ({ ...s, connected: false }));
    es.addEventListener("transaction", (evt: MessageEvent) => {
      try {
        const ev: LiveEvent = JSON.parse((evt as any).data);
        setState(s => ({
          ...s,
          connected: true,
          events: [ev, ...s.events].slice(0, maxBuffer),
          alerts: ev.alerted ? [ev, ...s.alerts].slice(0, maxBuffer) : s.alerts,
        }));
      } catch (e) {
        // ignore malformed lines
      }
    });
    return () => { es.close(); };
  }, [maxBuffer]);

  return state;
}
