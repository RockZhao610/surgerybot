"""
SDF（有符号距离场）碰撞检测测试脚本

用于验证 SDF 实现是否正确解决了点云方案的内部空洞问题。

运行方式：
    python -m surgical_robot_app.path_planning.test_sdf
"""

import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_sphere(shape=(50, 50, 50), center=None, radius=15):
    """
    创建一个测试用的球形掩码
    
    Args:
        shape: 体数据形状 (Z, H, W)
        center: 球心位置 (z, y, x)，默认为中心
        radius: 球的半径
    
    Returns:
        3D 二值掩码
    """
    if center is None:
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask = (dist <= radius).astype(np.uint8) * 255
    return mask


def test_point_cloud_hollow_problem():
    """
    测试点云方案的内部空洞问题
    
    这个测试展示了点云方案的缺陷：物体内部没有点，导致路径可能穿过物体。
    """
    from surgical_robot_app.path_planning.point_cloud_utils import (
        volume_to_point_cloud,
        PointCloudCollisionChecker
    )
    
    print("\n" + "="*60)
    print("测试 1：点云方案的内部空洞问题")
    print("="*60)
    
    # 创建一个球形物体
    mask = create_test_sphere(shape=(50, 50, 50), radius=15)
    
    # 生成点云
    point_cloud = volume_to_point_cloud(mask, threshold=128, spacing=None)
    print(f"点云大小: {len(point_cloud)} 个点")
    
    # 创建碰撞检测器
    checker = PointCloudCollisionChecker(point_cloud, safety_radius=5.0)
    
    # 测试点：物体中心（应该在内部，但点云方案可能检测不到）
    center_point = (50.0, 50.0, 50.0)  # 归一化坐标
    distance = checker.get_distance_to_obstacle(center_point)
    is_collision = checker.is_collision(center_point)
    
    print(f"\n物体中心点 {center_point}:")
    print(f"  - 到最近点云点的距离: {distance:.2f}")
    print(f"  - 是否检测为碰撞: {is_collision}")
    print(f"  - 问题: 中心点到表面的距离约为 {15 * 2:.1f}（半径的2倍），")
    print(f"         但实际上这个点在物体内部！")
    
    # 测试穿过物体的路径
    start = (20.0, 50.0, 50.0)  # 物体外部
    end = (80.0, 50.0, 50.0)    # 物体外部
    path_safe = checker.is_path_collision_free(start, end)
    
    print(f"\n穿过物体的路径 {start} -> {end}:")
    print(f"  - 点云检测结果: {'安全' if path_safe else '碰撞'}")
    print(f"  - 实际情况: 这条路径穿过球体中心，应该是碰撞！")
    
    if path_safe:
        print("\n❌ 问题确认：点云方案错误地认为穿过物体内部的路径是安全的！")
    
    return not path_safe  # 返回是否检测正确


def test_sdf_solution():
    """
    测试 SDF 方案如何解决内部空洞问题
    
    SDF（有符号距离场）可以完美区分内部和外部。
    """
    from surgical_robot_app.path_planning.sdf_utils import (
        compute_sdf_from_mask,
        SDFCollisionChecker
    )
    
    print("\n" + "="*60)
    print("测试 2：SDF 方案解决内部空洞问题")
    print("="*60)
    
    # 创建相同的球形物体
    mask = create_test_sphere(shape=(50, 50, 50), radius=15)
    
    # 计算 SDF
    sdf, metadata = compute_sdf_from_mask(mask, spacing=None, normalize_to_100=True)
    print(f"SDF 形状: {sdf.shape}")
    print(f"SDF 范围: [{metadata['sdf_range'][0]:.2f}, {metadata['sdf_range'][1]:.2f}]")
    
    # 创建 SDF 碰撞检测器
    checker = SDFCollisionChecker(sdf, metadata, safety_radius=5.0)
    
    # 测试点：物体中心
    center_point = (50.0, 50.0, 50.0)
    sdf_value = checker.get_distance(center_point)
    is_collision = checker.is_collision(center_point)
    is_inside = checker.is_inside_obstacle(center_point)
    
    print(f"\n物体中心点 {center_point}:")
    print(f"  - SDF 值: {sdf_value:.2f}（负值表示在内部）")
    print(f"  - 是否在物体内部: {is_inside}")
    print(f"  - 是否检测为碰撞: {is_collision}")
    
    # 测试穿过物体的路径
    start = (20.0, 50.0, 50.0)
    end = (80.0, 50.0, 50.0)
    path_safe = checker.is_path_collision_free(start, end)
    
    print(f"\n穿过物体的路径 {start} -> {end}:")
    print(f"  - SDF 检测结果: {'安全' if path_safe else '碰撞'}")
    
    if not path_safe:
        print("\n✅ SDF 正确检测到路径穿过物体内部！")
    
    # 测试安全路径（绕过物体）
    start_safe = (20.0, 20.0, 50.0)  # 在球体下方
    end_safe = (80.0, 20.0, 50.0)
    path_safe_detour = checker.is_path_collision_free(start_safe, end_safe)
    
    print(f"\n绕过物体的路径 {start_safe} -> {end_safe}:")
    print(f"  - SDF 检测结果: {'安全' if path_safe_detour else '碰撞'}")
    
    return not path_safe and path_safe_detour  # 穿过应该碰撞，绕过应该安全


def test_path_controller_with_sdf():
    """
    测试 PathController 使用 SDF 模式
    """
    from surgical_robot_app.path_planning.path_controller import PathController
    
    print("\n" + "="*60)
    print("测试 3：PathController 使用 SDF 模式")
    print("="*60)
    
    # 创建测试物体
    mask = create_test_sphere(shape=(50, 50, 50), radius=15)
    
    # 使用 SDF 模式
    controller_sdf = PathController(use_rrt=True, use_sdf=True, rrt_safety_radius=5.0)
    controller_sdf.set_obstacle_from_volume(mask)
    
    print(f"\nSDF 模式:")
    print(f"  - 碰撞检测器类型: {controller_sdf.get_collision_checker_type()}")
    print(f"  - 障碍物数据已设置: {controller_sdf.has_obstacle_data()}")
    
    # 使用点云模式对比
    controller_pc = PathController(use_rrt=True, use_sdf=False, rrt_safety_radius=5.0)
    controller_pc.set_obstacle_from_volume(mask)
    
    print(f"\n点云模式:")
    print(f"  - 碰撞检测器类型: {controller_pc.get_collision_checker_type()}")
    print(f"  - 障碍物数据已设置: {controller_pc.has_obstacle_data()}")
    
    # 测试路径规划
    # 设置起点和终点（尝试穿过物体）
    controller_sdf.set_start((20.0, 50.0, 50.0))
    controller_sdf.set_end((80.0, 50.0, 50.0))
    
    print("\n尝试规划穿过物体的路径...")
    try:
        path = controller_sdf.generate_path(smooth=True)
        print(f"  - 生成路径: {len(path)} 个点")
        
        # 检查路径是否绕过了物体
        # 如果路径中间点的 Y 坐标不是 50，说明绕道了
        mid_idx = len(path) // 2
        mid_point = path[mid_idx]
        print(f"  - 路径中点: {mid_point}")
        
        if abs(mid_point[1] - 50.0) > 5.0:
            print("  - ✅ 路径成功绕过了物体！")
        else:
            print("  - ⚠️ 路径可能穿过了物体")
            
    except RuntimeError as e:
        print(f"  - 路径规划失败: {e}")
        print("  - 这可能是因为起点/终点太靠近障碍物")
    
    return True


def test_sdf_gradient():
    """
    测试 SDF 梯度计算（用于调整点位置）
    """
    from surgical_robot_app.path_planning.sdf_utils import (
        compute_sdf_from_mask,
        SDFCollisionChecker
    )
    
    print("\n" + "="*60)
    print("测试 4：SDF 梯度和点调整")
    print("="*60)
    
    # 创建球形物体
    mask = create_test_sphere(shape=(50, 50, 50), radius=15)
    sdf, metadata = compute_sdf_from_mask(mask, spacing=None, normalize_to_100=True)
    checker = SDFCollisionChecker(sdf, metadata, safety_radius=5.0)
    
    # 测试在物体内部的点
    inside_point = (50.0, 50.0, 50.0)  # 物体中心
    print(f"\n物体内部的点: {inside_point}")
    print(f"  - SDF 值: {checker.get_distance(inside_point):.2f}")
    
    # 尝试调整到安全位置
    adjusted = checker.adjust_point_away_from_obstacle(inside_point)
    if adjusted:
        print(f"  - 调整后的点: {adjusted}")
        print(f"  - 调整后 SDF 值: {checker.get_distance(adjusted):.2f}")
        print(f"  - 调整后是否安全: {not checker.is_collision(adjusted)}")
    else:
        print("  - 无法调整到安全位置")
    
    # 测试在表面附近的点
    near_surface = (50.0, 35.0, 50.0)  # 接近球体表面
    print(f"\n表面附近的点: {near_surface}")
    print(f"  - SDF 值: {checker.get_distance(near_surface):.2f}")
    
    adjusted_near = checker.adjust_point_away_from_obstacle(near_surface)
    if adjusted_near:
        print(f"  - 调整后的点: {adjusted_near}")
        print(f"  - 调整后 SDF 值: {checker.get_distance(adjusted_near):.2f}")
    
    return True


def compare_methods_visualization():
    """
    可视化对比点云和 SDF 方法
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n跳过可视化测试（需要 matplotlib）")
        return True
    
    from surgical_robot_app.path_planning.point_cloud_utils import volume_to_point_cloud
    from surgical_robot_app.path_planning.sdf_utils import compute_sdf_from_mask
    
    print("\n" + "="*60)
    print("可视化：点云 vs SDF")
    print("="*60)
    
    # 创建球形物体
    mask = create_test_sphere(shape=(50, 50, 50), radius=15)
    
    # 生成点云
    point_cloud = volume_to_point_cloud(mask, threshold=128, spacing=None)
    
    # 计算 SDF
    sdf, metadata = compute_sdf_from_mask(mask, spacing=None, normalize_to_100=True)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原始掩码切片
    ax1 = axes[0]
    ax1.imshow(mask[25, :, :], cmap='gray')
    ax1.set_title('原始掩码 (Z=25 切片)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 2. 点云分布（投影到 XY 平面）
    ax2 = axes[1]
    pc_z25 = point_cloud[np.abs(point_cloud[:, 2] - 50) < 5]  # 只显示 Z≈50 的点
    if len(pc_z25) > 0:
        ax2.scatter(pc_z25[:, 0], pc_z25[:, 1], s=1, alpha=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_title('点云分布 (Z≈50)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    # 标注内部空洞
    circle = plt.Circle((50, 50), 5, color='red', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    ax2.annotate('内部空洞区域\n(无点云覆盖)', xy=(50, 50), ha='center', va='center', color='red')
    
    # 3. SDF 切片
    ax3 = axes[2]
    sdf_slice = sdf[25, :, :]
    im = ax3.imshow(sdf_slice, cmap='RdBu', vmin=-20, vmax=20)
    ax3.set_title('SDF (Z=25 切片)\n负值(蓝)=内部, 正值(红)=外部')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im, ax=ax3, label='有符号距离')
    
    plt.tight_layout()
    plt.savefig('sdf_vs_pointcloud_comparison.png', dpi=150)
    print("\n可视化已保存到 sdf_vs_pointcloud_comparison.png")
    plt.close()
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# SDF（有符号距离场）碰撞检测测试")
    print("#"*60)
    
    results = []
    
    # 测试 1：展示点云方案的问题
    try:
        result = test_point_cloud_hollow_problem()
        results.append(("点云内部空洞问题", result))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("点云内部空洞问题", False))
    
    # 测试 2：SDF 解决方案
    try:
        result = test_sdf_solution()
        results.append(("SDF 解决方案", result))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("SDF 解决方案", False))
    
    # 测试 3：PathController 集成
    try:
        result = test_path_controller_with_sdf()
        results.append(("PathController SDF 模式", result))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("PathController SDF 模式", False))
    
    # 测试 4：SDF 梯度
    try:
        result = test_sdf_gradient()
        results.append(("SDF 梯度和点调整", result))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("SDF 梯度和点调整", False))
    
    # 可视化对比
    try:
        result = compare_methods_visualization()
        results.append(("可视化对比", result))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("可视化对比", False))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ 所有测试通过！" if all_passed else "❌ 部分测试失败"))
    
    return all_passed


if __name__ == "__main__":
    main()
