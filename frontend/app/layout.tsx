import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Apni Awaaz",
  description: "Type or tap a reply. Hear it in a clone of your voice.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
