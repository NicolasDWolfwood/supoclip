import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // Skip TypeScript errors during builds for now
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
