const rl = @import("raylib");
const std = @import("std");
const fonts = @import("fonts.zig");

const screen_width = 64;
const screen_height = 32;
const display_size = screen_width * screen_height;
const max_program_size = 4096 - 512;

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
        0x0 => switch (@as(u8, @truncate(instructionCode))) {
            0xE0 => .{ .ClearScreen = {} },
            0xEE => .{ .Return = {} },
            else => error.InvalidInstruction,
        },
        0x1 => .{ .Jump = .{
            .dest = @truncate(instructionCode),
        } },
        0x2 => .{ .Call = .{
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
    Return,
    Jump: struct { dest: u12 },
    Call: struct { dest: u12 },
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

    fn load_font(self: *Emulator, font: [80]u8) void {
        std.mem.copyForwards(u8, self.memory[0x050..0x100], &font);
    }

    fn loadProgram(self: *Emulator, program: []const u8) !void {
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
    fn execute_instruction(self: *Emulator, instruction: Instruction) !void {
        // std.debug.print("{}\n", .{instruction});
        switch (instruction) {
            .ClearScreen => @memset(&self.display, 0),
            .Return => {
                self.pc = self.stack.popOrNull() orelse return error.NoContainingFrame;
            },
            .Jump => |instr| {
                self.pc = instr.dest;
            },
            .Call => |instr| {
                try self.stack.append(self.pc); // already advanced
                self.pc = @as(u16, instr.dest);
            },
            .Set => |instr| self.registers[instr.reg] = instr.val,
            .Add => |instr| self.registers[instr.reg] +|= instr.val,
            .SetIndex => |instr| self.index = instr.val,
            .Draw => |instr| {
                self.drawSprite(
                    self.index,
                    self.registers[instr.regX],
                    self.registers[instr.regY],
                    instr.height,
                );
            },
            // else => {
            //    error.UnimplementedInstruction
            // },
        }
    }

    /// Decodes and executes the instruction at `self.pc`, advancing the program counter as applicable.
    fn step(self: *Emulator) !void {
        const instruction_code: u16 = (@as(u16, self.memory[self.pc]) << 8) + self.memory[self.pc + 1];
        // std.debug.print("{d}: {X}\n", .{ self.pc, instruction_code });
        self.pc += 2;
        try self.execute_instruction(try decode_instruction(instruction_code));
    }

    /// Retrieves the Nth bit of memory.
    fn getBitValue(self: *Emulator, n: u16) u1 {
        const byte = n >> 3;
        const offset: u3 = @truncate(n % 8);
        const mask = @as(u8, 0b10000000) >> offset;
        return @intFromBool((self.memory[byte] & mask) > 0);
    }

    /// Draws the 8xN pixel sprite at sprite_ptr onto the screen at (x_pos % 64, y_pos % 64). Sprite drawing will not wrap, except for the origin.
    /// After this operation, VF will be 1 if a pixel was turned off this way.
    /// Otherwise, it will be 0.
    fn drawSprite(self: *Emulator, sprite_ptr: u16, x_pos: u8, y_pos: u8, height: u8) void {
        var y: u8 = y_pos % screen_height;
        var overwrote = false;
        for (0..height) |row| {
            var x: u8 = x_pos % screen_width;
            for (0..8) |col| {
                const offset: u16 = @as(u16, @truncate(row)) * 8 + @as(u16, @truncate(col));
                const val = self.getBitValue(
                    sprite_ptr * 8 + offset,
                );
                // std.debug.print("{X}", .{val});
                if (val > 0) {
                    const flag = self.flipPixel(x, y);
                    overwrote = flag or overwrote;
                }
                x +|= 1;
                if (x >= screen_width) {
                    break;
                }
            }
            // std.debug.print("\n", .{});
            y +|= 1;
            if (y >= screen_height) {
                break;
            }
        }
        // Set VF
        self.registers[0xF] = @intFromBool(overwrote);
    }

    /// Inverts the pixel given by x and y.
    /// Returns true iff a pixel was turned off this way.
    /// Assumes x and y are within display bounds.
    fn flipPixel(self: *Emulator, x: u8, y: u8) bool {
        const pixel = @as(u32, y) * screen_width + x;
        const flag = (self.display[pixel] > 0);
        self.display[pixel] = ~self.display[pixel];
        return flag;
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

    fn readProgramFromFile(self: *Emulator, file: std.fs.File) !void {
        var content_buf: [3584]u8 = undefined;
        const program_length = try file.readAll(&content_buf);
        try self.loadProgram(content_buf[0..program_length]);
    }
};
/// Returns an owned slice of bytes representing the memory as a hex dump.
/// The caller is responsible for freeing the memory with allocator.free(result).
fn formatHexDump(mem: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const row_width = 4;
    const block_width = 2;
    var output_buffer = std.ArrayList(u8).init(allocator);
    errdefer output_buffer.deinit();
    const output_writer = output_buffer.writer();
    for (0..(mem.len / (row_width * block_width))) |row| {
        for (0..row_width) |block| {
            const pos = row * row_width * block_width + block * block_width;
            var block_data: [block_width]u8 = undefined;
            std.mem.copyForwards(u8, &block_data, mem[pos .. pos + block_width]);
            _ = try output_writer.write(&std.fmt.bytesToHex(block_data, std.fmt.Case.upper));
            if (block < row_width - 1) {
                _ = try output_writer.write(" ");
            } else {
                _ = try output_writer.write("\n");
            }
        }
    }
    return output_buffer.toOwnedSlice();
}

const target_fps = 60;
const target_frame_time = 1.0 / @as(f64, @floatFromInt(target_fps));

const instructions_per_second = 700;
// const instructions_per_second = 100 * std.math.pow(u64, 10, 6);
const seconds_per_instruction: f64 = 1.0 /
    @as(f64, @floatFromInt(instructions_per_second));

pub fn main() !void {
    const window_width = 1600;
    const window_height = 960;

    // var prng = std.rand.DefaultPrng.init(0);
    // var rand = prng.random();

    rl.initWindow(window_width, window_height, "ZCrisp Emulator");
    defer rl.closeWindow();

    rl.setTargetFPS(target_fps);

    var recentFrameTimes: [20]f32 = [_]f32{0.0} ** 20;
    var frameClock: u8 = 0;
    var frameTimeBuf: [32]u8 = undefined;
    var recentFramesTotal: f32 = 0.0;
    var frameTimeText: []u8 = "";

    var stack_mem = [_]u8{0} ** 16;
    var stack_alloc = std.heap.FixedBufferAllocator.init(&stack_mem);
    var emulator = Emulator.new(stack_alloc.allocator());
    emulator.load_font(fonts.default);
    const screen = emulator.screenToTexture();

    const filename = "programs/ibm-logo.ch8";
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fs.realpath(filename, &path_buf);
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    try emulator.readProgramFromFile(file);

    std.debug.print("\n", .{});
    const hex_dump_size = 16000;
    var hexdump_buffer: [hex_dump_size]u8 = [_]u8{0} ** hex_dump_size;
    var hexdump_allocator = std.heap.FixedBufferAllocator.init(&hexdump_buffer);
    std.debug.print("{s}", .{try formatHexDump(emulator.memory[0x0200..0x0250], hexdump_allocator.allocator())});

    var seconds_to_simulate: f64 = 0.0;

    while (!rl.windowShouldClose()) {
        // Runs once each frame.
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.white);
        rl.drawText("Work in progress!", window_width / 2 - 100, 64, 20, rl.Color.black);

        frameClock +%= 1;
        const time = rl.getTime();
        seconds_to_simulate += rl.getFrameTime();
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
        std.debug.print("\x1B[2K\r", .{}); // Clear the line
        if (seconds_to_simulate > 2 * target_frame_time) {
            // out of sync
            std.debug.print("(Overloaded by {d:.2} ms) ", .{seconds_to_simulate - target_frame_time});
            seconds_to_simulate = target_frame_time;
        }
        std.debug.print("Simulating {d:.2} ms ({d:.0} instructions)", .{ seconds_to_simulate * 1000, seconds_to_simulate / seconds_per_instruction });
        while (seconds_to_simulate > seconds_per_instruction) {
            try emulator.step();
            seconds_to_simulate -= seconds_per_instruction;
        }

        // const randX = rand.intRangeLessThan(u8, 0, screen_width);
        // const randY = rand.intRangeLessThan(u8, 0, screen_height);
        // std.debug.print("Drawing sprite at ({d}, {d})\n", .{ randX, randY });
        // _ = emulator.drawPixel(true, randX, randY);
        // emulator.execute_instruction(.{ .ClearScreen = {} });
        // emulator.drawSprite(0x050 * 8, 28, 13, 5);
        // std.debug.print("\x1B[2K\r", .{}); // Clear the line

        const scale = 16.0;
        rl.updateTexture(screen, &emulator.display);
        rl.drawTextureEx(screen, rl.Vector2{
            .x = (window_width - screen_width * scale) / 2,
            .y = (window_height - screen_height * scale) / 2,
        }, 0.0, scale, rl.Color.dark_green);
    }
    std.debug.print("\n", .{});
    // const hex_dump_size = 16000;
    // var hexdump_buffer: [hex_dump_size]u8 = [_]u8{0} ** hex_dump_size;
    // var hexdump_allocator = std.heap.FixedBufferAllocator.init(&hexdump_buffer);
    // std.debug.print("{s}", .{try formatHexDump(emulator.memory[0x0200..0x0250], hexdump_allocator.allocator())});
}

const expect = std.testing.expect;
test "program initialization" {
    var emulator = Emulator.new(std.testing.allocator);
    const ones: [0x200]u8 = [_]u8{0xFF} ** 0x200;
    try emulator.loadProgram(&ones);
    try expect(emulator.memory[0x000] == 0x00);
    try expect(emulator.memory[0x100] == 0x00);
    try expect(emulator.memory[0x200] == 0xFF);
    try expect(emulator.memory[0x201] == 0xFF);
    try expect(emulator.memory[0x3FF] == 0xFF);
    try expect(emulator.memory[0x400] == 0x00);

    const tooLong = [_]u8{0x00} ** 0x2000;
    try expect(emulator.loadProgram(&tooLong) == error.OutofMemory);
}

test "timer tick" {
    var emulator = Emulator.new(std.testing.allocator);
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
    var emulator = Emulator.new(std.testing.allocator);
    try expect(false == emulator.flipPixel(true, 20, 20));
}

test "get bit value" {
    var emulator = Emulator.new(std.testing.allocator);
    const ones: [0x200]u8 = [_]u8{0xFF} ** 0x200;
    try emulator.loadProgram(&ones);
    try expect(emulator.getBitValue(0x250 * 8 + 1) == 1);
    try expect(emulator.getBitValue(0x60 * 8 + 7) == 0);
}
