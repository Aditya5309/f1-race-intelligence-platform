import type { Metadata } from "next";
import { Bodoni_Moda, Oswald, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Didone-style display serif — headlines, italic for emphasized words.
const display = Bodoni_Moda({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["600", "700"],
  style: ["normal", "italic"],
});

// Condensed uppercase — navigation/labels.
const label = Oswald({
  variable: "--font-label",
  subsets: ["latin"],
  weight: ["500", "600"],
});

// Tabular monospace — probabilities, dates, counts.
const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "F1 Race Winner Prediction",
  description: "Pre-race win-probability predictions, race by race.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${label.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[var(--color-page-bg)] text-[var(--color-text)]">
        {children}
      </body>
    </html>
  );
}
