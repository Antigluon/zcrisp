const rl = @import("raylib");
const std = @import("std");

const display_size = 64 * 32 / 8;

const Opcode = enum(u4) {
    ClearScreen = 0x0,
    Jump = 0x1,
    Set = 0x6,
    Add = 0x7,
    SetIndex = 0xA,
    Draw = 0xD,
};

fn decode_instruction(instructionCode: u16) !Instruction {
    const opcode: u4 = @truncate(instructionCode >> 12);
    return switch (opcode) {
        0x0 => .{ .ClearScreen = {} },
        0x1 => .{ .Jump = .{ .dest = @truncate(instructionCode) } },
        0x6 => .{ .Set = .{ .reg = @truncate(instructionCode >> 8), .val = @truncate(instructionCode) } },
        0x7 => .{ .Add = .{ .reg = @truncate(instructionCode >> 8), .val = @truncate(instructionCode) } },
        0xA => .{ .SetIndex = .{ .val = @truncate(instructionCode) } },
        0xD => .{ .Draw = .{ .regX = @truncate(instructionCode >> 8), .regY = @truncate(instructionCode >> 4), .height = @truncate(instructionCode) } },
        else => error.InvalidInput,
    };
}

const Instruction = union(enum) {
    ClearScreen,
    Jump: struct { dest: u12 },
    Set: struct { reg: u4, val: u8 },
    Add: struct { reg: u4, val: u8 },
    SetIndex: struct { val: u12 },
    Draw: struct { regX: u4, regY: u4, height: u4 },
};

const Emulator = struct {
    memory: [4096]u8,
    display: [display_size]u8,
    pc: u16,
    index: u16,
    stack: std.ArrayList(u16),
    registers: [16]u8,
    delay_timer: u8,
    sound_timer: u8,

    fn new() Emulator {
        var stack_mem = [_]u8{0} ** 16;
        var stack_alloc = std.heap.FixedBufferAllocator.init(&stack_mem);
        return Emulator{
            .memory = [_]u8{0x00} ** 4096,
            .display = [_]u8{0x00} ** display_size,
            .registers = [_]u8{0x00} ** 16,
            .pc = 0x200,
            .index = 0,
            .delay_timer = 0,
            .sound_timer = 0,
            .stack = std.ArrayList(u16).init(stack_alloc.allocator()),
        };
    }

    fn load_program(self: *Emulator, program: []const u8) !void {
        if (program.len > self.memory.len - 0x200) {
            return error.OutofMemory;
        }
        std.mem.copyForwards(u8, self.memory[(0x200)..], program);
    }

    fn tick_timers(self: *Emulator) void {
        self.delay_timer -|= 1;
        self.sound_timer -|= 1;
    }

    fn execute_instruction(self: *Emulator, instr: Instruction) void {
        self;
        instr;
        @panic("@ the disco");
    }
};

pub fn main() !void {
    const screenWidth = 1280;
    const screenHeight = 640;

    rl.initWindow(screenWidth, screenHeight, "ZCrisp Emulator");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    var recentFrameTimes: [20]f32 = [_]f32{0.0} ** 20;
    var frameClock: u8 = 0;
    var frameTimeBuf: [32]u8 = undefined;
    var recentFramesTotal: f32 = 0.0;
    var frameTimeText: []u8 = "";

    var emulator = Emulator.new();

    while (!rl.windowShouldClose()) {
        // Runs once each frame.
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.white);
        rl.drawText("Work in progress!", screenWidth / 2 - 100, screenHeight / 2 - 20, 20, rl.Color.black);

        frameClock +%= 1;
        const time = rl.getTime();
        const deltaTime = rl.getFrameTime() * 1000; // milliseconds
        recentFramesTotal += deltaTime;
        recentFramesTotal -= recentFrameTimes[frameClock % 20];
        recentFrameTimes[frameClock % 20] = deltaTime;
        if (time > 0.6 and frameClock % 8 == 0) {
            frameTimeText = try std.fmt.bufPrint(&frameTimeBuf, "Frame time: {d:.2} ms\n", .{recentFramesTotal / 20});
            frameTimeText[frameTimeText.len - 1] = 0;
        }
        rl.drawText(@ptrCast(frameTimeText), 10, 10, 20, rl.Color.dark_gray);

        // Emulator logic

        emulator.tick_timers();
    }
}

const expect = std.testing.expect;
test "program initialization" {
    var emulator = Emulator.new();
    const ones: [0x200]u8 = [_]u8{0xFF} ** 0x200;
    try emulator.load_program(&ones);
    try expect(emulator.memory[0x000] == 0x00);
    try expect(emulator.memory[0x100] == 0x00);
    try expect(emulator.memory[0x200] == 0xFF);
    try expect(emulator.memory[0x201] == 0xFF);
    try expect(emulator.memory[0x3FF] == 0xFF);
    try expect(emulator.memory[0x400] == 0x00);

    const tooLong = [_]u8{0x00} ** 0x2000;
    try expect(emulator.load_program(&tooLong) == error.OutofMemory);
}

test "timer tick" {
    var emulator = Emulator.new();
    emulator.delay_timer = 10;
    emulator.sound_timer = 5;
    emulator.tick_timers();
    try expect(emulator.delay_timer == 9);
    try expect(emulator.sound_timer == 4);
    emulator.delay_timer = 1;
    emulator.sound_timer = 0;
    emulator.tick_timers();
    try expect(emulator.delay_timer == 0);
    try expect(emulator.sound_timer == 0);
    emulator.tick_timers();
    try expect(emulator.delay_timer == 0);
    try expect(emulator.sound_timer == 0);
}

test "instruction decode" {
    const clear = try decode_instruction(0x0E0);
    try expect(clear.ClearScreen == {});

    const jump = try decode_instruction(0x1200);
    try expect(jump.Jump.dest == 0x200);

    const set = try decode_instruction(0x64F0);
    try expect(set.Set.reg == 0x4);
    try expect(set.Set.val == 0xF0);

    const add = try decode_instruction(0x780F);
    try expect(add.Add.reg == 0x8);
    try expect(add.Add.val == 0x0F);

    const draw = try decode_instruction(0xD238);
    try expect(draw.Draw.regX == 0x2);
    try expect(draw.Draw.regY == 0x3);
    try expect(draw.Draw.height == 0x8);
}
