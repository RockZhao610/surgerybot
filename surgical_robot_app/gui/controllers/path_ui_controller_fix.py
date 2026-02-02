# 临时修复文件 - 用于修复缩进问题
# 读取原文件
with open('surgical_robot_app/gui/controllers/path_ui_controller.py', 'r') as f:
    lines = f.readlines()

# 修复第142行（索引141）
if len(lines) > 141:
    line = lines[141]
    if line.strip().startswith('actor = actors.GetNextItem()'):
        # 确保有正确的缩进（20个空格，与while True对齐）
        lines[141] = '                    actor = actors.GetNextItem()\n'

# 修复第153行（索引152）
if len(lines) > 152:
    line = lines[152]
    if 'has_model = True' in line:
        # 确保有正确的缩进（36个空格，在if num_points > 100块内）
        lines[152] = '                                    has_model = True\n'

# 写回文件
with open('surgical_robot_app/gui/controllers/path_ui_controller.py', 'w') as f:
    f.writelines(lines)

print("修复完成")

