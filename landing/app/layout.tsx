import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "CephaloAI — AI-Powered Cephalometric Analysis",
  description:
    "Automatic landmark detection, instant measurements, and comprehensive clinical reports powered by deep learning. Trusted by orthodontists worldwide.",
  keywords: ["cephalometric", "AI", "orthodontics", "landmark detection", "clinical analysis"],
  openGraph: {
    title: "CephaloAI — AI-Powered Cephalometric Analysis",
    description: "Precision cephalometric analysis in seconds.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
