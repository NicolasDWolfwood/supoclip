import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  devIndicators: false,
  // Skip TypeScript errors during builds for now
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
