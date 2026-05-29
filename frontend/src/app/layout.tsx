import "./globals.css";
import Sidebar from "@/components/Sidebar";
import { ReactNode } from "react";

export const metadata = {
  title: "GRAPH.AI",
  description: "Graph-based Risk Analysis & Pattern Hunting",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-bg text-text">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 px-10 py-8 overflow-x-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
