import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { nextCookies } from "better-auth/next-js";
import prisma from "./prisma";

const baseURL =
  process.env.BETTER_AUTH_URL ||
  process.env.NEXT_PUBLIC_APP_URL ||
  "http://localhost:3000";

const trustedOrigins = new Set<string>([
  baseURL,
  "http://localhost:3000",
  "http://127.0.0.1:3000",
  "http://[::1]:3000",
]);

const extraTrustedOrigins = (process.env.BETTER_AUTH_TRUSTED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

for (const origin of extraTrustedOrigins) {
  trustedOrigins.add(origin);
}

export const auth = betterAuth({
  baseURL,
  trustedOrigins: [...trustedOrigins],
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),
  emailAndPassword: {
    enabled: true,
  },
  plugins: [
    nextCookies(), // Enable Next.js cookie handling
  ],
});

export type Session = typeof auth.$Infer.Session;
