import type { Metadata } from "next";
import { Newsreader, Outfit } from "next/font/google";
import { LanguageProvider } from "@/components/Language";
import "./globals.css";

const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", display: "swap" });
const sans = Outfit({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

export const metadata: Metadata = {
  title: "Pretty Little Pictures",
  description: "An independent study of RGB-to-thermal generation when paired data are scarce.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable}`}>
      <body><LanguageProvider>{children}</LanguageProvider></body>
    </html>
  );
}
