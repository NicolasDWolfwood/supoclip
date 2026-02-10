import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SupoClip",
  description: "Turn long videos into viral-ready shorts.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
