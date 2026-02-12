import { createAuthClient } from "better-auth/react";

const resolvedBaseUrl =
  process.env.NEXT_PUBLIC_APP_URL ||
  (typeof window !== "undefined" ? window.location.origin : undefined);

export const authClient = createAuthClient({
  baseURL: resolvedBaseUrl,
});

export const {
  signIn,
  signOut,
  signUp,
  useSession,
} = authClient;
