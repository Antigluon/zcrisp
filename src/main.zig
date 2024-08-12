const rl = @import("raylib");
const std = @import("std");
const fonts = @import("fonts");

const screen_width = 64;
const screen_height = 32;
const display_size = screen_width * screen_height;

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
        0x1 => .{ .Jump = .{
            .dest = @truncate(instructionCode),
        } },
        0x6 => .{ .Set = .{
            .reg = @truncate(instructionCode >> 8),
            .val = @truncate(instructionCode),
        } },
        0x7 => .{ .Add = .{
            .reg = @truncate(instructionCode >> 8),
            .val = @truncate(instructionCode),
        } },
        0xA => .{ .SetIndex = .{
            .val = @truncate(instructionCode),
        } },
        0xD => .{ .Draw = .{
            .regX = @truncate(instructionCode >> 8),
            .regY = @truncate(instructionCode >> 4),
            .height = @truncate(instructionCode),
        } },
        else => error.InvalidInstruction,
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

    fn new(stack_allocator: std.mem.Allocator) Emulator {
        return Emulator{
            .memory = [_]u8{0x00} ** 4096,
            .display = [_]u8{0x00} ** display_size,
            .registers = [_]u8{0x00} ** 16,
            .pc = 0x200,
            .index = 0,
            .delay_timer = 0,
            .sound_timer = 0,
            .stack = std.ArrayList(u16).init(stack_allocator),
        };
    }

    fn load_font(self: *Emulator, font: [64]u8) void {
        std.mem.copyForwards(u8, self.memory[0x50..0x9F], font);
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

    /// Executes a decoded instruction.
    fn execute_instruction(self: *Emulator, instruction: Instruction) void {
        // @panic("@ the disco");
        switch (instruction) {
            .ClearScreen => @memset(self.display, 0),
            .Jump => |instr| self.pc = instr.dest,
            .Set => |instr| self.registers[instr.reg] = instr.val,
            .Add => |instr| self.registers[instr.reg] +|= instr.val,
            .SetIndex => |instr| self.index = instr.val,
            .Draw => |instr| self.drawSprite(
                self.memory[self.index],
                self.registers[instr.regX],
                self.registers[instr.regY],
                instr.height,
            ),
        }
    }

    /// Decodes and executes the instruction at `self.pc`, advancing the program counter as applicable.
    fn step(self: *Emulator) !void {
        const instruction_code = self.memory[self.pc];
        self.pc += 2;
        self.execute_instruction(try decode_instruction(instruction_code));
    }

    fn drawSprite(self: *Emulator, sprite_ptr: u16, x_pos: u8, y_pos: u8, height: u8) void {
        var y: u8 = y_pos % screen_height;
        for (0..height) |row| {
            for (0..8) |_| {
                var x = x_pos % screen_width;
                _ = (y) * screen_width + x;
                _ = self.memory(sprite_ptr) + row;
                x += 1;
            }
            y += 1;
        }
    }

    /// Writes `state` to the pixel given by x and y.
    /// Sets VF <- 1 if a pixel was turned off this way.
    /// Assumes x and y are within display bounds.
    fn drawPixel(self: *Emulator, state: bool, x: u8, y: u8) void {
        const pixel = @as(u32, y) * screen_width + x;
        if (self.display[pixel] > 0 and !state) {
            // Set VF
            self.registers[0xF] = 1;
        }
        self.display[pixel] = 0xFF * @as(u8, @intFromBool(state));
    }

    fn screenToTexture(self: *Emulator) rl.Texture {
        const screen = rl.Image{
            .data = &self.display,
            .width = screen_width,
            .height = screen_height,
            .mipmaps = 1,
            .format = rl.PixelFormat.pixelformat_uncompressed_grayscale,
        };
        return rl.Texture.fromImage(screen);
    }
};

pub fn main() !void {
    const window_width = 1600;
    const window_height = 960;

    var prng = std.rand.DefaultPrng.init(0);
    var rand = prng.random();

    rl.initWindow(window_width, window_height, "ZCrisp Emulator");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    var recentFrameTimes: [20]f32 = [_]f32{0.0} ** 20;
    var frameClock: u8 = 0;
    var frameTimeBuf: [32]u8 = undefined;
    var recentFramesTotal: f32 = 0.0;
    var frameTimeText: []u8 = "";

    var stack_mem = [_]u8{0} ** 16;
    var stack_alloc = std.heap.FixedBufferAllocator.init(&stack_mem);
    var emulator = Emulator.new(stack_alloc.allocator());
    const screen = emulator.screenToTexture();

    while (!rl.windowShouldClose()) {
        // Runs once each frame.
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.white);
        rl.drawText("Work in progress!", window_width / 2 - 100, 64, 20, rl.Color.black);

        frameClock +%= 1;
        const time = rl.getTime();
        const deltaTime = rl.getFrameTime() * 1000; // milliseconds
        recentFramesTotal += deltaTime;
        recentFramesTotal -= recentFrameTimes[frameClock % 20];
        recentFrameTimes[frameClock % 20] = deltaTime;
        if (time > 0.6 and frameClock % 8 == 0) {
            frameTimeText = try std.fmt.bufPrint(&frameTimeBuf, "Frame time: {d:.3} ms\n", .{recentFramesTotal / 20});
            frameTimeText[frameTimeText.len - 1] = 0;
        }
        rl.drawText(@ptrCast(frameTimeText), 10, 10, 20, rl.Color.dark_gray);

        // Emulator logic

        emulator.tick_timers();

        const randX = rand.intRangeLessThan(u8, 0, screen_width);
        const randY = rand.intRangeLessThan(u8, 0, screen_height);
        emulator.drawPixel(true, randX, randY);
        std.debug.print("\x1B[2K\r", .{}); // Clear the line
        std.debug.print("Drawing pixel at ({d}, {d})", .{ randX, randY });

        const scale = 16.0;
        rl.updateTexture(screen, &emulator.display);
        rl.drawTextureEx(screen, rl.Vector2{
            .x = (window_width - screen_width * scale) / 2,
            .y = (window_height - screen_height * scale) / 2,
        }, 0.0, scale, rl.Color.dark_green);
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

test "draw pixel" {
    var emulator = Emulator.new();
    emulator.drawPixel(true, 20, 20);
}
