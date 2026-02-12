import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import prisma from "@/lib/prisma";

const SUPPORTED_TRANSCRIPTION_PROVIDERS = new Set(["local", "assemblyai"]);
const SUPPORTED_AI_PROVIDERS = new Set(["openai", "google", "anthropic"]);

// GET /api/preferences - Get user preferences
export async function GET() {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: {
        default_font_family: true,
        default_font_size: true,
        default_font_color: true,
        default_transitions_enabled: true,
        default_transcription_provider: true,
        default_ai_provider: true,
      },
    });

    if (!user) {
      return NextResponse.json(
        { error: "User not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      fontFamily: user.default_font_family || "TikTokSans-Regular",
      fontSize: user.default_font_size || 24,
      fontColor: user.default_font_color || "#FFFFFF",
      transitionsEnabled: user.default_transitions_enabled ?? false,
      transcriptionProvider: user.default_transcription_provider || "local",
      aiProvider: user.default_ai_provider || "openai",
    });
  } catch (error) {
    console.error("Error fetching preferences:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

// PATCH /api/preferences - Update user preferences
export async function PATCH(request: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const body = await request.json();
    const {
      fontFamily,
      fontSize,
      fontColor,
      transitionsEnabled,
      transcriptionProvider,
      aiProvider,
    } = body;

    // Validate inputs
    if (fontFamily !== undefined && typeof fontFamily !== "string") {
      return NextResponse.json(
        { error: "Invalid fontFamily" },
        { status: 400 }
      );
    }

    if (fontSize !== undefined && (typeof fontSize !== "number" || fontSize < 12 || fontSize > 48)) {
      return NextResponse.json(
        { error: "Invalid fontSize (must be between 12 and 48)" },
        { status: 400 }
      );
    }

    if (fontColor !== undefined && !/^#[0-9A-Fa-f]{6}$/.test(fontColor)) {
      return NextResponse.json(
        { error: "Invalid fontColor (must be hex format like #FFFFFF)" },
        { status: 400 }
      );
    }

    if (transitionsEnabled !== undefined && typeof transitionsEnabled !== "boolean") {
      return NextResponse.json(
        { error: "Invalid transitionsEnabled" },
        { status: 400 }
      );
    }

    if (
      transcriptionProvider !== undefined &&
      (typeof transcriptionProvider !== "string" ||
        !SUPPORTED_TRANSCRIPTION_PROVIDERS.has(transcriptionProvider))
    ) {
      return NextResponse.json(
        { error: "Invalid transcriptionProvider (must be local or assemblyai)" },
        { status: 400 }
      );
    }

    if (
      aiProvider !== undefined &&
      (typeof aiProvider !== "string" || !SUPPORTED_AI_PROVIDERS.has(aiProvider))
    ) {
      return NextResponse.json(
        { error: "Invalid aiProvider (must be openai, google, or anthropic)" },
        { status: 400 }
      );
    }

    const updatedUser = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        ...(fontFamily !== undefined && { default_font_family: fontFamily }),
        ...(fontSize !== undefined && { default_font_size: fontSize }),
        ...(fontColor !== undefined && { default_font_color: fontColor }),
        ...(transitionsEnabled !== undefined && { default_transitions_enabled: transitionsEnabled }),
        ...(transcriptionProvider !== undefined && { default_transcription_provider: transcriptionProvider }),
        ...(aiProvider !== undefined && { default_ai_provider: aiProvider }),
      },
      select: {
        default_font_family: true,
        default_font_size: true,
        default_font_color: true,
        default_transitions_enabled: true,
        default_transcription_provider: true,
        default_ai_provider: true,
      },
    });

    return NextResponse.json({
      fontFamily: updatedUser.default_font_family,
      fontSize: updatedUser.default_font_size,
      fontColor: updatedUser.default_font_color,
      transitionsEnabled: updatedUser.default_transitions_enabled ?? false,
      transcriptionProvider: updatedUser.default_transcription_provider || "local",
      aiProvider: updatedUser.default_ai_provider || "openai",
    });
  } catch (error) {
    console.error("Error updating preferences:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
