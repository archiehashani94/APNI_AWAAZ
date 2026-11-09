import type { Metadata } from "next";
import { Figtree, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const display = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument",
  display: "swap",
});

const body = Figtree({
  subsets: ["latin"],
  variable: "--font-figtree",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Apni Awaaz",
  description: "Type or tap a reply. Hear it in a clone of your voice.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${body.variable} ${mono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
