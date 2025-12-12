import math

def matmul3(A, B):
    """3x3 矩阵乘法"""
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def normalize_angle(angle):
    """归一化角度到 [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def euler_to_matrix_scenic(yaw, pitch, roll):
    """Scenic 欧拉角转旋转矩阵"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    
    Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    
    return matmul3(matmul3(Rz, Ry), Rx)

def matrix_to_euler_xyz(R):
    """旋转矩阵转 MuJoCo XYZ 外旋欧拉角"""
    if abs(R[2][0]) < 0.999999:
        ry = -math.asin(R[2][0])
        rx = math.atan2(R[2][1]/math.cos(ry), R[2][2]/math.cos(ry))
        rz = math.atan2(R[1][0]/math.cos(ry), R[0][0]/math.cos(ry))
    else:
        ry = -math.asin(R[2][0])
        rx = 0
        rz = math.atan2(-R[0][1], R[1][1])
    
    return [normalize_angle(rx), normalize_angle(ry), normalize_angle(rz)]

def combine_scenic_to_mujoco_euler(orientation, rotationOffset=None):
    """
    将 Scenic orientation 转换为 MuJoCo euler
    包含坐标系转换：Scenic -> MuJoCo 需要绕 Z 轴旋转 +90°
    """
    # 1. 获取基础朝向
    yaw1 = getattr(orientation, "yaw", 0.0)
    pitch1 = getattr(orientation, "pitch", 0.0)
    roll1 = getattr(orientation, "roll", 0.0)
    
    R1 = euler_to_matrix_scenic(yaw1, pitch1, roll1)
    
    # 2. 应用 rotationOffset（如果有）
    if rotationOffset is not None and rotationOffset not in [(0, 0, 0), [0, 0, 0]]:
        if hasattr(rotationOffset, "yaw"):
            yaw2 = rotationOffset.yaw
            pitch2 = rotationOffset.pitch
            roll2 = rotationOffset.roll
        elif isinstance(rotationOffset, (tuple, list)) and len(rotationOffset) == 3:
            yaw2, pitch2, roll2 = rotationOffset
        else:
            raise TypeError(f"Unsupported rotationOffset type: {type(rotationOffset)}")
        
        R2 = euler_to_matrix_scenic(yaw2, pitch2, roll2)
        R_total = matmul3(R1, R2)
    else:
        R_total = R1
    
    # Scenic 和 MuJoCo 的 X/Y 轴定义不同，需要绕 Z 轴旋转 +90°
    Rz_correction = euler_to_matrix_scenic(math.pi / 2, 0, 0)
    R_mujoco = matmul3(R_total, Rz_correction)
    
    # 4. 转换为 MuJoCo XYZ 欧拉角
    euler_xyz = matrix_to_euler_xyz(R_mujoco)
    
    return euler_xyz