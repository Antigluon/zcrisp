const rl = @import("raylib");
const std = @import("std");

const Emulator = struct {
    memory: [4096]u8,
    display: std.bit_set.ArrayBitSet(u8, 64 * 32),
    pc: u16,
    index: u16,
    stack: std.ArrayList(u16),
    delay_timer: u8,
    sound_timer: u8,
    registers: [16]u8,
};

pub fn main() !void {
    const screenWidth = 800;
    const screenHeight = 450;

    rl.initWindow(screenWidth, screenHeight, "ZCrisp Emulator");
    defer rl.closeWindow();

    while (!rl.windowShouldClose()) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.white);
        rl.drawText("Work in progress!", screenWidth / 2 - 100, screenHeight / 2 - 20, 20, rl.Color.black);
    }
}
