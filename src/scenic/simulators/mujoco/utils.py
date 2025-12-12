# scenic/simulators/mujoco/utils.py - 修复版本

import math
from scenic.core.vectors import Orientation, Vector
from scenic.core.shapes import BoxShape, CylinderShape, SpheroidShape, ConeShape, MeshShape

# ============================================================
# 配置和常量
# ============================================================

class MuJoCoConversionConfig:
    """MuJoCo 转换配置"""
    def __init__(self):
        # 调试选项
        self.debug = False
        self.verbose = False
        
        # 场景设置
        self.add_ground_plane = False  # 默认不添加额外 ground
        self.ground_z_offset = -1.0    # 如果添加 ground，放在这个高度
        
        # 默认物理参数
        self.default_density = 1000.0  # kg/m³
        
        # 特殊对象处理
        self.floor_as_plane = True    # 是否将 Floor 转换为 plane
        self.auto_detect_floor = True  # 自动检测 Floor 对象
        
        # 相机和光照
        self.add_default_camera = True
        self.add_lighting = True
        self.add_debug_axes = False

# 全局配置实例
MUJOCO_CONFIG = MuJoCoConversionConfig()

# ============================================================
# 基础工具函数
# ============================================================

def _matmul3(A, B):
    """3x3 矩阵乘法"""
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _transform_vertex(v, T):
    """用变换矩阵转换顶点"""
    return [
        T[0][0]*v[0] + T[0][1]*v[1] + T[0][2]*v[2],
        T[1][0]*v[0] + T[1][1]*v[1] + T[1][2]*v[2],
        T[2][0]*v[0] + T[2][1]*v[1] + T[2][2]*v[2]
    ]

def get_object_type(obj):
    """获取对象类型名称"""
    obj_class = type(obj)
    # 尝试获取最具体的类型名
    if hasattr(obj_class, '__qualname__'):
        return obj_class.__qualname__
    return obj_class.__name__

# ============================================================
# 坐标转换 - 修复版
# ============================================================

def scenic_position_to_mujoco(position):
    """
    Scenic (X右,Y前,Z上) -> MuJoCo (X前,Y左,Z上)
    
    修复：正确的坐标转换
    MuJoCo: X前, Y左, Z上
    Scenic: X右, Y前, Z上
    
    转换关系：
    MuJoCo_X = Scenic_Y
    MuJoCo_Y = -Scenic_X
    MuJoCo_Z = Scenic_Z
    """
    return [position.y, -position.x, position.z]


def scenic_vector_to_mujoco(vector):
    """转换向量（速度、偏移等）"""
    return [vector.y, -vector.x, vector.z]

import math

def euler_to_matrix_scenic(yaw, pitch, roll):
    """
    Scenic 使用 Z-Y-X 内旋顺序
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    
    # Z 旋转矩阵
    Rz = [
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ]
    
    # Y 旋转矩阵
    Ry = [
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ]
    
    # X 旋转矩阵
    Rx = [
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ]
    
    # 组合: R = Rz * Ry * Rx (内旋，从右向左)
    def matmul(A, B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    
    R = matmul(matmul(Rz, Ry), Rx)
    return R


def matrix_to_euler_xyz(R):
    """
    从旋转矩阵提取 XYZ 外旋欧拉角（MuJoCo 格式）
    """
    # XYZ 外旋: R = Rz(rz) * Ry(ry) * Rx(rx)
    # 
    # R = [[ cy*cz,  -sz,  sy*cz],
    #      [ cy*sz,   cz,  sy*sz],
    #      [   -sy,    0,     cy]]
    
    if abs(R[2][0]) < 0.999999:
        ry = -math.asin(R[2][0])
        rx = math.atan2(R[2][1]/math.cos(ry), R[2][2]/math.cos(ry))
        rz = math.atan2(R[1][0]/math.cos(ry), R[0][0]/math.cos(ry))
    else:
        # 万向节锁
        ry = -math.asin(R[2][0])
        rx = 0
        rz = math.atan2(-R[0][1], R[1][1])
    
    return [rx, ry, rz]


def scenic_orientation_to_mujoco(orientation):
    """
    将 Scenic Orientation 转换为 MuJoCo euler
    """
    yaw = orientation.yaw
    pitch = orientation.pitch
    roll = orientation.roll
    
    # 1. 构建 Scenic 的旋转矩阵
    R = euler_to_matrix_scenic(yaw, pitch, roll)
    
    # 2. 从矩阵提取 MuJoCo 的 XYZ 欧拉角
    euler_xyz = matrix_to_euler_xyz(R)
    
    return euler_xyz

# ============================================================
# 形状处理 - 修复版
# ============================================================

def scenic_dimensions_to_mujoco(width, length, height, shape_type):
    """
    转换尺寸（修复版）
    
    注意：在 Scenic 中
    - width: X 方向（右）
    - length: Y 方向（前）
    - height: Z 方向（上）
    
    在 MuJoCo 中：
    - X: 前
    - Y: 左
    - Z: 上
    """
    if shape_type == 'box':
        # box size: [X半长, Y半长, Z半长]
        # MuJoCo X = Scenic Y, MuJoCo Y = Scenic X
        return [length/2, width/2, height/2]
    elif shape_type == 'cylinder':
        # cylinder size: [半径, 半高]
        # 假设圆柱体沿 Z 轴
        radius = max(width, length) / 2
        return [radius, height/2]
    elif shape_type == 'ellipsoid':
        # ellipsoid size: [X半轴, Y半轴, Z半轴]
        return [length/2, width/2, height/2]
    elif shape_type == 'sphere':
        # sphere size: [半径]
        radius = max(width, length, height) / 2
        return [radius]
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

def scenic_color_to_rgba(color, default_color=None):
    """转换颜色"""
    if color is None:
        if default_color is None:
            return [0.5, 0.5, 0.5, 1.0]
        color = default_color
    
    if hasattr(color, 'rgb'):
        rgb = color.rgb
    elif isinstance(color, (list, tuple)):
        rgb = color
    else:
        try:
            rgb = list(color)
        except:
            return [0.5, 0.5, 0.5, 1.0]
    
    if len(rgb) == 3:
        return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
    elif len(rgb) == 4:
        return [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(rgb[3])]
    else:
        return [0.5, 0.5, 0.5, 1.0]

# ============================================================
# Mesh 处理
# ============================================================

def transform_mesh_vertices(vertices, center_mesh=False):
    """转换 mesh 顶点"""
    T_coord = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    
    vertices_list = []
    for v in vertices:
        vertices_list.append([float(v[0]), float(v[1]), float(v[2])])
    
    center_offset = [0.0, 0.0, 0.0]
    
    if center_mesh and len(vertices_list) > 0:
        n = len(vertices_list)
        cx = sum(v[0] for v in vertices_list) / n
        cy = sum(v[1] for v in vertices_list) / n
        cz = sum(v[2] for v in vertices_list) / n
        
        if abs(cx) > 0.01 or abs(cy) > 0.01 or abs(cz) > 0.01:
            center_offset = [cx, cy, cz]
            vertices_list = [[v[0]-cx, v[1]-cy, v[2]-cz] for v in vertices_list]
    
    vertices_mujoco = [_transform_vertex(v, T_coord) for v in vertices_list]
    
    return vertices_mujoco, center_offset

def validate_mesh_data(vertices, faces, name="mesh"):
    """验证 mesh 数据"""
    try:
        vertices_len = len(vertices)
    except:
        return False, f"{name}: Invalid vertices"
    
    try:
        faces_len = len(faces)
    except:
        return False, f"{name}: Invalid faces"
    
    if vertices_len == 0:
        return False, f"{name}: No vertices"
    
    if faces_len == 0:
        return False, f"{name}: No faces"
    
    for i, v in enumerate(vertices):
        try:
            if len(v) != 3:
                return False, f"{name}: Vertex {i} has {len(v)} components"
        except:
            return False, f"{name}: Vertex {i} is invalid"
    
    max_vertex_idx = vertices_len - 1
    for i, f in enumerate(faces):
        try:
            f_len = len(f)
        except:
            return False, f"{name}: Face {i} is invalid"
        
        if f_len < 3:
            return False, f"{name}: Face {i} has {f_len} vertices"
        
        for j, idx in enumerate(f):
            idx_val = int(idx)
            if idx_val < 0 or idx_val > max_vertex_idx:
                return False, f"{name}: Face {i} vertex {j} out of range"
    
    return True, "Valid"

def fix_mesh_faces(faces):
    """修复面索引"""
    fixed_faces = []
    for face in faces:
        fixed_face = [int(idx) for idx in face]
        
        if len(fixed_face) != len(set(fixed_face)):
            seen = set()
            fixed_face = [idx for idx in fixed_face if not (idx in seen or seen.add(idx))]
        
        if len(fixed_face) >= 3:
            fixed_faces.append(fixed_face)
    
    return fixed_faces

def apply_mesh_initial_rotation(vertices, initial_rotation):
    """应用 mesh 初始旋转"""
    if initial_rotation is None:
        return vertices
    
    if isinstance(initial_rotation, Orientation):
        yaw = initial_rotation.yaw
        pitch = initial_rotation.pitch
        roll = initial_rotation.roll
    elif isinstance(initial_rotation, (list, tuple)) and len(initial_rotation) == 3:
        yaw, pitch, roll = initial_rotation
    else:
        return vertices
    
    R = euler_to_matrix_scenic(yaw, pitch, roll)
    vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in vertices]
    rotated_vertices = [_transform_vertex(v, R) for v in vertices_list]
    
    return rotated_vertices

def process_mesh_shape(shape, name, debug=False):
    """处理 Mesh/Cone Shape"""
    mesh = shape.mesh
    vertices_scenic = mesh.vertices
    faces_scenic = mesh.faces
    
    if debug:
        print(f"  Processing mesh: {len(vertices_scenic)} vertices, {len(faces_scenic)} faces")
    
    is_valid, error_msg = validate_mesh_data(vertices_scenic, faces_scenic, name)
    if not is_valid:
        raise ValueError(f"Invalid mesh data: {error_msg}")
    
    vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in vertices_scenic]
    
    if isinstance(shape, MeshShape):
        initial_rotation = getattr(shape, 'initial_rotation', None)
        if initial_rotation is not None:
            vertices_list = apply_mesh_initial_rotation(vertices_list, initial_rotation)
    
    center_mesh = isinstance(shape, ConeShape)
    vertices_mujoco, center_offset = transform_mesh_vertices(vertices_list, center_mesh)
    faces_fixed = fix_mesh_faces(faces_scenic)
    
    return vertices_mujoco, faces_fixed, center_offset

# ============================================================
# 物理属性
# ============================================================

def calculate_mass_from_density(density, shape, width, length, height):
    """计算质量"""
    if density is None or density <= 0:
        return None
    
    if isinstance(shape, BoxShape):
        volume = width * length * height
    elif isinstance(shape, CylinderShape):
        radius = max(width, length) / 2
        volume = math.pi * radius * radius * height
    elif isinstance(shape, SpheroidShape):
        a, b, c = width/2, length/2, height/2
        volume = (4/3) * math.pi * a * b * c
    elif isinstance(shape, ConeShape):
        radius = max(width, length) / 2
        volume = (1/3) * math.pi * radius * radius * height
    else:
        return None
    
    return density * volume

# ============================================================
# 对象创建 - 修复版
# ============================================================

def create_object_in_mujoco(mjcfRoot, obj, config=None):
    """
    将 Scenic Object 转换为 MuJoCo body 和 geom
    
    修复版：正确处理位置、偏移和坐标转换
    """
    if config is None:
        config = MUJOCO_CONFIG
    
    # 获取基本属性
    name = getattr(obj, "name", f"obj_{id(obj)}")
    obj_type = get_object_type(obj)
    shape = getattr(obj, "shape", None)
    
    if shape is None:
        raise ValueError(f"Object {name} ({obj_type}) has no shape")
    
    # 尺寸
    width = getattr(obj, "width", 1.0)
    length = getattr(obj, "length", 1.0)
    height = getattr(obj, "height", 1.0)
    
    # 位置和朝向（原始值）
    position = getattr(obj, "position", Vector(0, 0, 0))
    orientation = getattr(obj, "orientation", Orientation.fromEuler(0, 0, 0))
    
    # 偏移量
    base_offset = getattr(obj, "baseOffset", None)
    position_offset = getattr(obj, "positionOffset", None)
    rotation_offset = getattr(obj, "rotationOffset", None)
    
    # === 关键修复：正确处理偏移 ===
    # Scenic 的位置包含了 baseOffset，所以不需要减去
    # 只需要加上 positionOffset（如果有）
    final_position = position
    
    if position_offset is not None:
        if isinstance(position_offset, Vector):
            final_position = Vector(
                position.x + position_offset.x,
                position.y + position_offset.y,
                position.z + position_offset.z
            )
        elif isinstance(position_offset, (list, tuple)):
            final_position = Vector(
                position.x + position_offset[0],
                position.y + position_offset[1],
                position.z + position_offset[2]
            )
    
    # 处理旋转偏移
    final_orientation = orientation
    if rotation_offset is not None:
        if isinstance(rotation_offset, (list, tuple)) and len(rotation_offset) == 3:
            # rotation_offset 是 (yaw, pitch, roll)
            extra_orient = Orientation.fromEuler(
                rotation_offset[0],  # yaw
                rotation_offset[1],  # pitch
                rotation_offset[2]   # roll
            )
            final_orientation = orientation * extra_orient
    
    # 坐标转换
    pos_mujoco = scenic_position_to_mujoco(final_position)
    euler_mujoco = scenic_orientation_to_mujoco(final_orientation)
    
    # 物理属性
    density = getattr(obj, "density", None)
    if density is None:
        density = config.default_density
    
    mass = calculate_mass_from_density(density, shape, width, length, height)
    allow_collisions = getattr(obj, "allowCollisions", True)
    contact_tolerance = getattr(obj, "contactTolerance", 0.0001)
    
    # 视觉属性
    color = getattr(obj, "color", None)
    rgba = scenic_color_to_rgba(color, default_color=[0.5, 0.8, 0.5])
    render = getattr(obj, "render", True)
    
    # 运动属性
    velocity = getattr(obj, "velocity", Vector(0, 0, 0))
    angular_velocity = getattr(obj, "angularVelocity", Vector(0, 0, 0))
    has_initial_velocity = (velocity != Vector(0, 0, 0) or 
                           angular_velocity != Vector(0, 0, 0))
    
    # 调试输出
    if config.debug:
        print(f"\n[{obj_type}] {name}")
        if config.verbose:
            print(f"  Shape: {type(shape).__name__} (W={width:.2f}, L={length:.2f}, H={height:.2f})")
            print(f"  Scenic Pos: ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})")
            print(f"  MuJoCo Pos: ({pos_mujoco[0]:.2f}, {pos_mujoco[1]:.2f}, {pos_mujoco[2]:.2f})")
            if base_offset:
                print(f"  Base Offset: ({base_offset.x:.2f}, {base_offset.y:.2f}, {base_offset.z:.2f})")
            if position_offset:
                print(f"  Pos Offset: {position_offset}")
            if orientation.yaw != 0 or orientation.pitch != 0 or orientation.roll != 0:
                print(f"  Orientation: yaw={math.degrees(orientation.yaw):.1f}° "
                      f"pitch={math.degrees(orientation.pitch):.1f}° "
                      f"roll={math.degrees(orientation.roll):.1f}°")
    
    # 创建 body
    body = mjcfRoot.worldbody.add("body", name=name, pos=pos_mujoco, euler=euler_mujoco)
    
    if has_initial_velocity:
        body.add("freejoint", name=f"{name}_freejoint")
    
    # geom 参数
    geom_kwargs = {
        "rgba": rgba if render else [0, 0, 0, 0],
        "contype": 1 if allow_collisions else 0,
        "conaffinity": 1 if allow_collisions else 0,
    }
    
    if mass is not None and mass > 0:
        geom_kwargs["mass"] = mass
    
    if contact_tolerance != 0.0001:
        geom_kwargs["margin"] = contact_tolerance
    
    # 创建 geom
    if isinstance(shape, BoxShape):
        size = scenic_dimensions_to_mujoco(width, length, height, 'box')
        body.add("geom", type="box", size=size, **geom_kwargs)
    
    elif isinstance(shape, CylinderShape):
        size = scenic_dimensions_to_mujoco(width, length, height, 'cylinder')
        body.add("geom", type="cylinder", size=size, **geom_kwargs)
    
    elif isinstance(shape, SpheroidShape):
        if abs(width - length) < 1e-6 and abs(length - height) < 1e-6:
            size = scenic_dimensions_to_mujoco(width, length, height, 'sphere')
            body.add("geom", type="sphere", size=size, **geom_kwargs)
        else:
            size = scenic_dimensions_to_mujoco(width, length, height, 'ellipsoid')
            body.add("geom", type="ellipsoid", size=size, **geom_kwargs)
    
    elif isinstance(shape, (ConeShape, MeshShape)):
        vertices_mujoco, faces_fixed, center_offset = process_mesh_shape(
            shape, name, debug=config.verbose
        )
        
        # 处理mesh中心偏移
        if any(abs(c) > 0.01 for c in center_offset):
            # 将 Scenic 坐标系的偏移转换到 MuJoCo 坐标系
            offset_scenic = Vector(center_offset[0], center_offset[1], center_offset[2])
            offset_mujoco = scenic_vector_to_mujoco(offset_scenic)
            body.pos = [
                body.pos[0] + offset_mujoco[0],
                body.pos[1] + offset_mujoco[1],
                body.pos[2] + offset_mujoco[2]
            ]
        
        mesh_name = f"mesh_{name}"
        vertex_flat = [coord for vertex in vertices_mujoco for coord in vertex]
        face_flat = [idx for face in faces_fixed for idx in face]
        
        mjcfRoot.asset.add("mesh", name=mesh_name, vertex=vertex_flat, face=face_flat)
        body.add("geom", type="mesh", mesh=mesh_name, **geom_kwargs)
    
    else:
        raise ValueError(f"Unsupported shape type: {type(shape)}")
    
    if has_initial_velocity:
        body._scenic_initial_velocity = scenic_vector_to_mujoco(velocity)
        body._scenic_initial_angvel = scenic_vector_to_mujoco(angular_velocity)
    
    return body

# ============================================================
# 场景创建
# ============================================================

def create_scene_in_mujoco(mjcfRoot, scenic_objects, config=None):
    """将 Scenic 场景转换为 MuJoCo"""
    if config is None:
        config = MUJOCO_CONFIG
    
    if config.debug:
        print(f"\n{'='*70}")
        print(f"Creating MuJoCo Scene from Scenic")
        print(f"Objects: {len(scenic_objects)}")
        print(f"{'='*70}")
    
    bodies = []
    has_floor = False
    
    # 第一遍：检测是否有 Floor
    if config.auto_detect_floor:
        for obj in scenic_objects:
            obj_type = get_object_type(obj)
            if 'floor' in obj_type.lower():
                has_floor = True
                break
    
    # 添加 ground plane（如果需要）
    if config.add_ground_plane and not has_floor:
        ground = mjcfRoot.worldbody.add("body", name="ground", 
                                       pos=[0, 0, config.ground_z_offset])
        ground.add("geom", type="plane", size=[100, 100, 0.1],
                  rgba=[0.3, 0.3, 0.3, 1], contype=1, conaffinity=1)
        if config.debug:
            print(f"Added ground plane at Z={config.ground_z_offset}")
    
    # 添加光照
    if config.add_lighting:
        mjcfRoot.worldbody.add("light", name="ambient", directional=False,
                              diffuse=[0.5, 0.5, 0.5], pos=[0, 0, 5], castshadow=False)
        mjcfRoot.worldbody.add("light", name="sun", directional=True,
                              diffuse=[0.6, 0.6, 0.6], pos=[0, 0, 10], 
                              dir=[0, 0, -1], castshadow=True)
    
    # 添加默认相机
    if config.add_default_camera:
        mjcfRoot.worldbody.add("camera", name="scenic_view",
                              pos=[0, -8, 6], quat=[0.92, 0.38, 0, 0], mode="fixed")
    
    # 添加调试坐标轴
    if config.add_debug_axes:
        add_coordinate_axes(mjcfRoot, length=2.0)
    
    # 创建所有对象
    for i, obj in enumerate(scenic_objects):
        try:
            body = create_object_in_mujoco(mjcfRoot, obj, config=config)
            bodies.append(body)
        except Exception as e:
            obj_type = get_object_type(obj)
            obj_name = getattr(obj, 'name', f'obj_{i}')
            print(f"ERROR: Failed to create {obj_type} '{obj_name}': {e}")
            if config.verbose:
                import traceback
                traceback.print_exc()
    
    if config.debug:
        print(f"\nCreated {len(bodies)}/{len(scenic_objects)} objects successfully")
        print(f"{'='*70}\n")
    
    return bodies

def add_coordinate_axes(mjcfRoot, length=1.0, pos=None):
    """添加调试坐标轴"""
    if pos is None:
        pos = [0, 0, 0]
    axes_body = mjcfRoot.worldbody.add("body", name="debug_axes", pos=pos)
    
    for axis, color, target in [
        ('x', [1,0,0,0.8], [length,0,0]),
        ('y', [0,1,0,0.8], [0,length,0]),
        ('z', [0,0,1,0.8], [0,0,length])
    ]:
        axes_body.add("geom", type="cylinder", fromto=[0,0,0] + target,
                     size=[0.01], rgba=color, contype=0, conaffinity=0)
        axes_body.add("geom", type="sphere", pos=target, size=[0.03],
                     rgba=[color[0],color[1],color[2],1], contype=0, conaffinity=0)

# ============================================================
# 便捷接口
# ============================================================

def convert_scenic_to_mujoco(scene, debug=False, verbose=False, **kwargs):
    """
    便捷接口：将 Scenic 场景转换为 MuJoCo
    
    Args:
        scene: Scenic Scene 对象
        debug: 是否打印调试信息
        verbose: 是否打印详细信息
        **kwargs: 其他配置选项
    
    Returns:
        (mjcfRoot, bodies): MuJoCo 根节点和创建的 body 列表
    """
    from dm_control import mjcf
    
    # 创建配置
    config = MuJoCoConversionConfig()
    config.debug = debug
    config.verbose = verbose
    
    # 应用自定义配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 创建 MuJoCo 模型
    mjcfRoot = mjcf.RootElement()
    
    # 转换场景
    bodies = create_scene_in_mujoco(mjcfRoot, scene.objects, config=config)
    
    return mjcfRoot, bodies