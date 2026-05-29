"use client";

import { create } from "zustand";

// One selectionStore for the whole app: when the user clicks "Investigate"
// on the Overview, Case/Geo/Report all pick up the same `selected` payload.

export interface SelectionState {
  selectedTransactionId: string | null;
  selectedPayload: any | null;
  setSelected: (txId: string, payload: any) => void;
  clearSelection: () => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedTransactionId: null,
  selectedPayload: null,
  setSelected: (txId, payload) => set({ selectedTransactionId: txId, selectedPayload: payload }),
  clearSelection: () => set({ selectedTransactionId: null, selectedPayload: null }),
}));
