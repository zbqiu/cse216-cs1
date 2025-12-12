
"""Simulator interface for Mujoco."""

try:
    import mujoco, mujoco.viewer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "mujoco is required. Please install the 'mujoco' package."
    ) from e

try:
    from dm_control import mjcf, mujoco
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Deepmind dm_control is required. Please install the 'dm_control' package."
    ) from e

import scenic
import os, math, time
from scenic.core.simulators import Simulation, Simulator, SimulationCreationError
from scenic.simulators.mujoco.orientation import combine_scenic_to_mujoco_euler
from scenic.simulators.mujoco.physics_env import PhysicsEnv

class MujocoSimulator(Simulator):

    def __init__(self):
        super().__init__()
    
    def createSimulation(self, scene, debugMode=True, **kwargs):
        return MujocoSimulation(scene, debugMode, **kwargs)
        

class MujocoSimulation(Simulation):

    def __init__(self, scene, debugMode=True, addGround=False, xmlTemplatePath=None, 
                 applyJvrcModifications=False, jvrcEnv=None, jvrcPolicy=None, **kwargs):
        
        # 保存 JvrcWalkEnv 和 policy（如果提供）
        self.jvrcEnv = jvrcEnv
        self.jvrcPolicy = jvrcPolicy
        self.jvrcObs = None  # 当前观测
        self.first_fall_step = None  # 记录第一次摔倒的步数
        
        if xmlTemplatePath:
            try:
                
                # 如果需要，应用 JVRC builder 的修改
                if applyJvrcModifications:
                    print("applyJvrcModifications=True")
                    self.applyJvrcModifications = applyJvrcModifications
                    path_to_xml = '/tmp/mjcf-export/jvrc_walk/jvrc.xml'
                    print(f"Checking if {path_to_xml} exists: {os.path.exists(path_to_xml)}")
                    if not os.path.exists(path_to_xml):
                        print(f"Building model from {xmlTemplatePath} with {len(scene.objects)} terrain objects...")
                        self.build_model = mjcf.from_path(xmlTemplatePath)
                        export_dir = os.path.dirname(path_to_xml)
                        # 传入 scene 参数
                        self._apply_jvrc_modifications(export_dir, scene)
                    else:
                        print(f"Using cached XML (terrain objects will be missing!)")
                    
                    self.mjModel = mujoco.MjModel.from_xml_path(path_to_xml)
                    self.mjData = mujoco.MjData(self.mjModel)
                    print("0000=1")
                else:
                    print("0000")
                    self.mjcfRoot = mjcf.from_path(xmlTemplatePath)     

            except Exception as e:
                raise SimulationCreationError("Failed to create mjcf.RootElement from xml template")
        else:
            # construct a root node
            self.mjcfRoot = mjcf.RootElement(model="Scenic-Mujoco")
            self.mjcfRoot.compiler.angle = "radian"
            self.mjcfRoot.asset.add('texture', name="grid", type="2d", builtin="checker", rgb1=[.1, .2, .3], rgb2=[.4, .5, .6], width=512, height=512)
            self.mjcfRoot.asset.add('material', name="grid", texture="grid", texrepeat="8 8", reflectance=".2")
            if addGround:
                self.mjcfRoot.worldbody.add('geom',type="plane", name="default_ground", pos=[0, 0, 0], size=[1, 1, 0.1])
        
        self.debugMode = debugMode
        self._paused = True
        self._step_by_step = True  # do pause every timestep 
        self._step_once = False
        self._user_closed = False
        self.viewer = None
        

        kwargs.setdefault("maxSteps", 1000000)
        kwargs.setdefault("name", "test")
        kwargs.setdefault("timestep", 1)
        
        # 移除自定义参数，避免传给父类
        kwargs.pop('createJvrcEnv', None)
        kwargs.pop('xmlTemplatePath', None)
        
        super().__init__(scene, **kwargs)

    def _apply_jvrc_modifications(self, export_path, scene):        
        # 定义关节列表
        WAIST_JOINTS = ['WAIST_Y', 'WAIST_P', 'WAIST_R']
        HEAD_JOINTS = ['NECK_Y', 'NECK_R', 'NECK_P']
        HAND_JOINTS = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
                       'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']
        ARM_JOINTS = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_SHOULDER_Y', 'R_ELBOW_P', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
                      'L_SHOULDER_P', 'L_SHOULDER_R', 'L_SHOULDER_Y', 'L_ELBOW_P', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']
        LEG_JOINTS = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
                      'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']
        
        # 1. 设置基本参数
        self.build_model.model = 'jvrc'
        self.build_model.size.njmax = -1
        self.build_model.size.nconmax = -1
        self.build_model.statistic.meansize = 0.1
        self.build_model.statistic.meanmass = 2
        
        # 2. 修改天空盒为白色
        for tx in self.build_model.asset.texture:
            if tx.type == "skybox":
                tx.rgb1 = '1 1 1'
                tx.rgb2 = '1 1 1'
        
        # 3. 移除所有碰撞规则
        self.build_model.contact.remove()
        
        # 4. 只保留腿部执行器
        for mot in self.build_model.actuator.motor:
            if mot.joint.name not in LEG_JOINTS:
                mot.remove()
        
        # 5. 移除不需要的关节
        for joint in WAIST_JOINTS + HEAD_JOINTS + HAND_JOINTS + ARM_JOINTS:
            try:
                self.build_model.find('joint', joint).remove()
            except:
                pass  # 如果关节不存在，跳过
        
        # 6. 移除等式约束
        self.build_model.equality.remove()
        
        # 7. 固定手臂姿势
        arm_bodies = {
            "R_SHOULDER_P_S": [0, -0.052, 0], 
            "R_SHOULDER_R_S": [-0.17, 0, 0], 
            "R_ELBOW_P_S": [0, -0.524, 0],
            "L_SHOULDER_P_S": [0, -0.052, 0], 
            "L_SHOULDER_R_S": [0.17, 0, 0], 
            "L_ELBOW_P_S": [0, -0.524, 0],
        }
        for bname, euler in arm_bodies.items():
            try:
                self.build_model.find('body', bname).euler = euler
            except:
                pass
        
        # 8. 简化碰撞几何体
        collision_geoms = [
            'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
            'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
        ]
        
        for body in self.build_model.worldbody.find_all('body'):
            for idx, geom in enumerate(body.geom):
                geom.name = body.name + '-geom-' + repr(idx)
                if (geom.dclass.dclass == "collision"):
                    if body.name not in collision_geoms:
                        geom.remove()
        
        # 9. 设置碰撞组
        try:
            self.build_model.default.default['collision'].geom.group = 3
        except:
            pass
        
        # 10. 添加脚部碰撞几何体
        try:
            self.build_model.worldbody.find('body', 'R_ANKLE_P_S').add(
                'geom', dclass='collision', size='0.1 0.05 0.01', 
                pos='0.029 0 -0.09778', type='box'
            )
            self.build_model.worldbody.find('body', 'L_ANKLE_P_S').add(
                'geom', dclass='collision', size='0.1 0.05 0.01', 
                pos='0.029 0 -0.09778', type='box'
            )
        except:
            pass
        
        # 11. 排除膝盖-脚踝碰撞
        self.build_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
        self.build_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')
        
        # 12. 清理未使用的网格
        meshes = [g.mesh.name for g in self.build_model.find_all('geom') 
                  if g.type=='mesh' or g.type==None]
        for mesh in self.build_model.find_all('mesh'):
            if mesh.name not in meshes:
                mesh.remove()
        
        # 13. 修正传感器位置
        try:
            self.build_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
            self.build_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'
        except:
            pass

        # add box geoms (config 为空，不添加额外的 box)
        config = {}
        if 'boxes' in config and config['boxes']==True:
            for idx in range(20):
                name = 'box' + repr(idx+1).zfill(2)
                self.build_model.worldbody.add('body', name=name, pos=[0, 0, -0.2])
                self.build_model.find('body', name).add('geom',
                                                name=name,
                                                dclass='collision',
                                                group='0',
                                                size='1 1 0.1',
                                                type='box',
                                                material='')
        
        # 14. 重建地面（只在没有 Scenic 地形对象时创建默认平面）
        floor_geom = self.build_model.find('geom', 'floor')
        if floor_geom is not None:
            floor_material = floor_geom.material if hasattr(floor_geom, 'material') and floor_geom.material else None
            floor_geom.remove()
        else:
            floor_material = None
        
        # 如果没有 Scenic 地形对象，创建默认平面
        # 注意：Scenic objects 会在 setup() 中通过 createObjectInSimulator 自动添加
        if len(scene.objects) == 0:
            print("No Scenic terrain objects, creating default floor plane")
            self.build_model.worldbody.add('body', name='floor')
            if floor_material:
                self.build_model.find('body', 'floor').add(
                    'geom', name='floor', type="plane", size="0 0 0.25", material=floor_material
                )
            else:
                self.build_model.find('body', 'floor').add(
                    'geom', name='floor', type="plane", size="0 0 0.25", rgba="0.8 0.9 0.8 1"
                )
        else:
            print(f"{len(scene.objects)} Scenic terrain objects will be added in setup()")
        
        # 导出到 /tmp/mjcf-export/jvrc_walk/jvrc.xml
        mjcf.export_with_assets(self.build_model, out_dir=export_path, precision=5)
        path_to_xml = os.path.join(export_path, self.build_model.model + '.xml')
        print(f"JVRC modifications applied and exported to: {path_to_xml}")
    
    def _key_callback(self, keycode):
        """键盘回调处理"""
        if keycode == 32:  # SPACE - 单步前进 / 切换到单步模式
            if self._paused:
                # 暂停状态下，前进一步
                self._step_once = True
            else:
                # 连续运行中，切换到单步模式并暂停
                self._step_by_step = True
                self._paused = True
                print(f"[PAUSED] Step mode at step {self.currentTime} - press SPACE to step")
        elif keycode == 257 or keycode == 335:  # ENTER - 连续运行模式
            self._step_by_step = False
            self._paused = False
            print("[RUNNING] Continuous mode - press SPACE to pause")
        elif keycode == 81 or keycode == 113:  # Q - 退出
            self._user_closed = True
            if self.viewer:
                self.viewer.close()

    def setup(self):
        # 如果使用 JvrcWalkEnv
        if hasattr(self, 'applyJvrcModifications') and self.applyJvrcModifications:
            print("Using JvrcWalkEnv with Scenic terrain objects")
            
            # 1. 先创建 Scenic objects（需要 mjcfRoot）
            # 这时 self.build_model 已经在 _apply_jvrc_modifications 中创建好了
            print(f"Converting {len(self.scene.objects)} Scenic objects to MuJoCo geoms...")
            
            # 临时设置 mjcfRoot 以便 createObjectInSimulator 工作
            self.mjcfRoot = self.build_model
            
            # 调用父类的 object 创建逻辑（但不创建 model/data）
            self.agents = []
            self.objects = []
            for obj in self.scene.objects:
                self.createObjectInSimulator(obj)
                self.objects.append(obj)
            
            # 2. 导出修改后的 XML（包含 Scenic 添加的 geoms）
            #print(f"Exporting XML with Scenic objects...")
            xml_string = self.build_model.to_xml_string()
            export_path = '/tmp/mjcf-export/jvrc_walk'
            import os
            os.makedirs(export_path, exist_ok=True)
            xml_path = os.path.join(export_path, 'jvrc.xml')
            with open(xml_path, 'w') as f:
                f.write(xml_string)
            
            # 3. 创建 JvrcWalkEnv（会加载我们导出的 XML）
            if self.jvrcEnv is None:
                from envs.jvrc.jvrc_walk import JvrcWalkEnv
                self.jvrcEnv = JvrcWalkEnv()
                
                # 初始化环境
                obs = self.jvrcEnv.reset()
                self.jvrcEnv.task._goal_speed_ref = 0.35
                self.jvrcObs = self.jvrcEnv.get_obs()
        else:
            super().setup()
        
        # 如果使用 JvrcWalkEnv，使用它的 model/data
        if self.jvrcEnv is not None:
            self.mjModel = self.jvrcEnv.model
            self.mjData = self.jvrcEnv.data
            
            # 初始化观测
            self.jvrcObs = self.jvrcEnv.get_obs()
            
            # 创建 viewer
            if self.debugMode:
                self.viewer = mujoco.viewer.launch_passive(self.mjModel, self.mjData)
            
            # 初始化必要属性
            self.agents = []
            self.objects = self.scene.objects
            return
        
        # 原始逻辑（没有 JvrcWalkEnv 时）
        if self.debugMode:
            if not hasattr(self, 'applyJvrcModifications') or not self.applyJvrcModifications:
                xml_string = self.mjcfRoot.to_xml_string()
                with open("mod.xml", "w", encoding="utf-8") as f:
                    f.write(xml_string) 

                assets = self.mjcfRoot.get_assets()
                self.mjModel = mujoco.MjModel.from_xml_string(xml_string, assets)
                self.mjData = mujoco.MjData(self.mjModel)
            
            self.viewer = mujoco.viewer.launch_passive(self.mjModel, self.mjData)
        else:
            if not hasattr(self, 'applyJvrcModifications') or not self.applyJvrcModifications:
                xml_string = self.mjcfRoot.to_xml_string()
                with open("mod.xml", "w", encoding="utf-8") as f:
                    f.write(xml_string)                 
                self.physics = mujoco.Physics.from_xml_string(xml_string)
                pixels = self.physics.render()
                # TODO convert to image/video when necessary.

    def createObjectInSimulator(self, obj):
        """Create the given Scenic object in the MuJoCo simulator (dm_control.mjcf)."""
        
        # print("----------------------------------===========-------------------------")
        # print(f"\nObject: {obj}")
        # print("  Type:", type(obj))
        # print("  Attributes:")
        # for name, val in vars(obj).items():
        #     print(f"    {name} = {val}")
        # print("----------------------------------===========-------------------------")
        
        # print("self.agents[]:",self.agents)
        position = obj.position
        mj_position = position


        half_dimension = (obj.hl, obj.hw, obj.hh)

        orientation = obj.orientation
        rotationOffset = getattr(obj, "rotationOffset", None)
        mj_orientation = combine_scenic_to_mujoco_euler(orientation, rotationOffset)

        # color 
        if obj.color is not None:
            color = (obj.color[0], obj.color[1], obj.color[2], 1.0)
        else:
            color = (0, 1.0, 0.0, 1.0)  # default color green

        body_name = f"body_{id(obj)}"
        obj.mj_body_name = body_name
        
        # 🔧 临时禁用旋转，直接用 0 0 0 测试地形是否可见
        # body = self.mjcfRoot.worldbody.add('body', name=body_name, pos=mj_position, euler=mj_orientation)
        body = self.mjcfRoot.worldbody.add('body', name=body_name, pos=mj_position, euler="0 0 0")
        # if obj.behavior:
        #     body.add('freejoint', name=f"freejoint_{body_name}")

        # 根据 shape 类型创建 geom
        shape = obj.shape
        if isinstance(shape, scenic.core.shapes.BoxShape):
            half_dimension = (obj.hl, obj.hw, obj.hh)
            body.add('geom', type='box', size=half_dimension, rgba=color)
        
        elif isinstance(shape, scenic.core.shapes.CylinderShape):
            radius = getattr(obj, 'radius', max(obj.hl, obj.hw))
            height = obj.hh
            body.add('geom', type='cylinder', size=[radius, height], rgba=color)
        
        elif isinstance(shape, scenic.core.shapes.SpheroidShape):
            radius = getattr(obj, 'radius', max(obj.hl, obj.hw, obj.hh))
            body.add('geom', type='sphere', size=[radius], rgba=color)

        elif isinstance(shape, scenic.core.shapes.ConeShape):
            try:
                mesh = shape.mesh
                mesh_name = f"mesh_{id(obj)}"
                vertices_scenic = mesh.vertices
                faces_scenic = mesh.faces
                vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in vertices_scenic]

                faces_fixed = []
                for face in faces_scenic:
                    fixed_face = [int(idx) for idx in face]
                    
                    if len(fixed_face) != len(set(fixed_face)):
                        seen = set()
                        fixed_face = [idx for idx in fixed_face if not (idx in seen or seen.add(idx))]
                    
                    if len(fixed_face) >= 3:
                        faces_fixed.append(fixed_face)

                scale = [obj.radius, obj.radius, obj.height]
                vertices_scaled = [
                    [v[0]*scale[0], v[1]*scale[1], v[2]*scale[2]] for v in vertices_scenic
                ]

                vertex_flat = [coord for vertex in vertices_scaled for coord in vertex]
                face_flat = [idx for face in faces_fixed for idx in face]

                # add mesh
                self.mjcfRoot.asset.add(
                    "mesh",
                    vertex=vertex_flat,
                    face=face_flat,
                    name=mesh_name
                )
                body.add(
                    "geom",
                    type="ConeShape",
                    mesh=mesh_name, # use the asset by name
                )
            except Exception as e:
                raise SimulationCreationError("create Shape {} for object {} failed".format(type(shape), obj))
            
        elif isinstance(shape, scenic.core.shapes.MeshShape):
            try:
                # 方法 1: 从 shape.mesh 获取顶点和面
                if hasattr(shape, 'mesh') and shape.mesh is not None:
                    mesh = shape.mesh
                    mesh_name = f"mesh_{id(obj)}"
                    vertices_scenic = mesh.vertices
                    faces_scenic = mesh.faces
                    
                    # 获取物体的缩放信息
                    scale = [1.0, 1.0, 1.0]
                    if hasattr(obj, 'width'):
                        scale[0] = float(obj.width)
                    if hasattr(obj, 'length'):
                        scale[1] = float(obj.length)
                    if hasattr(obj, 'height'):
                        scale[2] = float(obj.height)
                    
                    # 如果有 meshScale 属性,使用它
                    if hasattr(obj, 'meshScale'):
                        mesh_scale = obj.meshScale
                        if isinstance(mesh_scale, (list, tuple)):
                            scale = [float(s) for s in mesh_scale]
                        else:
                            scale = [float(mesh_scale)] * 3
                    
                    # 转换顶点坐标并应用缩放
                    vertices_mujoco = []
                    for v in vertices_scenic:
                        # Scenic -> MuJoCo: (x,y,z) -> (y,-x,z)
                        # 同时应用缩放
                        x_mj = float(v[1]) * scale[1]  # Scenic的y -> MuJoCo的x
                        y_mj = float(-v[0]) * scale[0]  # Scenic的x -> MuJoCo的-y
                        z_mj = float(v[2]) * scale[2]   # Scenic的z -> MuJoCo的z
                        vertices_mujoco.extend([x_mj, y_mj, z_mj])
                    
                    # 处理面索引
                    faces_flat = []
                    for face in faces_scenic:
                        faces_flat.extend([int(idx) for idx in face])
                    
                    # 创建 mesh asset (不需要额外的scale参数,已经应用到顶点上了)
                    
                    self.mjcfRoot.asset.add('mesh',
                                        name=mesh_name,
                                        vertex=vertices_mujoco,
                                        face=faces_flat)
                    
                    # 创建 mesh geom
                    body.add('geom', type='mesh', mesh=mesh_name, rgba=color)
                # 方法 2: 从文件路径加载
                elif hasattr(shape, 'filename') and shape.filename is not None:
                    mesh_path = shape.filename
                    
                    if os.path.exists(mesh_path):
                        ext = os.path.splitext(mesh_path)[1].lower()
                        supported_formats = ['.stl', '.obj', '.dae', '.xml']
                        
                        if ext in supported_formats:
                            mesh_name = f"mesh_{id(obj)}"
                            
                            # 获取缩放信息
                            scale = None
                            if hasattr(obj, 'meshScale'):
                                mesh_scale = obj.meshScale
                                if isinstance(mesh_scale, (list, tuple)):
                                    scale = [float(s) for s in mesh_scale]
                                else:
                                    scale = [float(mesh_scale)] * 3
                            elif hasattr(obj, 'width') or hasattr(obj, 'length') or hasattr(obj, 'height'):
                                scale = [
                                    float(getattr(obj, 'length', 1.0)),
                                    float(getattr(obj, 'width', 1.0)),
                                    float(getattr(obj, 'height', 1.0))
                                ]
                            
                            # 添加 mesh 时需要考虑坐标转换
                            # MuJoCo的scale顺序也是 (x,y,z),需要对应转换
                            if scale is not None:
                                # 调整scale顺序以匹配坐标转换: Scenic(x,y,z) -> MuJoCo(y,-x,z)
                                scale_mj = [scale[1], scale[0], scale[2]]
                                self.mjcfRoot.asset.add('mesh',
                                                    name=mesh_name,
                                                    file=mesh_path,
                                                    scale=scale_mj)
                            else:
                                self.mjcfRoot.asset.add('mesh',
                                                    name=mesh_name,
                                                    file=mesh_path)
                            
                            body.add('geom', type='mesh', mesh=mesh_name, rgba=color)
                            
                        else:
                            raise SimulationCreationError("Unsupported mesh format {} . Create Shape {} for object {} failed".format(ext, type(shape), obj))
                    else:
                        raise SimulationCreationError("mesh file {} does not exists. Create Shape {} for object {} failed".format(mesh_path, type(shape), obj))

            
            except Exception as e:
                raise SimulationCreationError("create Shape {} for object {} failed".format(type(shape), obj))

        
        else:
            raise SimulationCreationError(
                "Unsupported shape type {} for object {}".format(type(shape), obj)
            )
        
        return body

    def step(self):
        """执行一个仿真步骤"""
        # 如果使用 JvrcWalkEnv，让它来控制仿真
        if self.jvrcEnv is not None and self.jvrcPolicy is not None:
            import torch
            import time
            
            # 🔧 实时限速：记录上一步的时间
            if not hasattr(self, '_last_step_time'):
                self._last_step_time = time.time()
            
            # 获取当前观测
            if self.jvrcObs is None:
                self.jvrcObs = self.jvrcEnv.get_obs()
            
            # 使用 policy 计算动作
            with torch.no_grad():
                action = self.jvrcPolicy.forward(
                    torch.tensor(self.jvrcObs, dtype=torch.float32),
                    deterministic=True
                ).detach().numpy()
            
            # 执行动作（env.step 会调用 mujoco.mj_step）
            self.jvrcObs, reward, done, info = self.jvrcEnv.step(action)
            
            # 如果倒下，记录第一次摔倒的步数
            if done:
                # 记录第一次摔倒
                if not hasattr(self, 'first_fall_step') or self.first_fall_step is None:
                    self.first_fall_step = self.currentTime
                
                # 重置环境
                self.jvrcObs = self.jvrcEnv.reset()
                self.jvrcEnv.task._goal_speed_ref = 0.35
                self.jvrcObs = self.jvrcEnv.get_obs()
            
            # 同步 viewer（env.data 已经被 env.step 更新了）
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
            
            # 每步应该是 0.025 秒（40Hz）
            elapsed = time.time() - self._last_step_time
            if elapsed < self.timestep:
                time.sleep(self.timestep - elapsed)
            self._last_step_time = time.time()
        
        elif self.debugMode:
            # 原始 MuJoCo 步进
            mujoco.mj_step(self.mjModel, self.mjData)
            if self.viewer is not None:
                self.viewer.sync()
        else:
            self.physics.step()

    def step0(self):
        """Run the simulation for one step and return the next trajectory element.

        Implemented by subclasses. This should cause the simulator to simulate physics
        for ``self.timestep`` seconds.
        """

        if self.debugMode:
            if self._user_closed or (self.viewer is not None and not self.viewer.is_running()):
                self.screen = "Dead"
                return
            
            # 暂停等待用户输入
            while self._paused and not self._step_once:
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                    time.sleep(0.001)
                else:
                    self.screen = "Dead"
                    return
            
            # 重置单步标志
            self._step_once = False
            
            # 执行物理步进
            mujoco.mj_step(self.mjModel, self.mjData)
            
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
            
            # 关键：如果是单步模式，执行完一步后重新暂停
            if self._step_by_step:
                self._paused = True
                print(f"[Step {self.currentTime + 1}] Press SPACE to step, or ENTER to continue")
        else:
            self.physics.step()

    
    def getProperties(self, obj, properties):
        """Read the values of the given properties of the object from the simulator.

        Implemented by subclasses.

        Args:
            obj (Object): Scenic object in question.
            properties (set): Set of names of properties to read from the simulator.
                It is safe to destructively iterate through the set if you want.

        Returns:
            A `dict` mapping each of the given properties to its current value.
        """
        result = {}
        for prop in properties:
            if prop == "position":
                result[prop] = obj.position
            else:
                result[prop] = getattr(obj, prop, None)
        return result
    
    def getBodyId(self, agent):
        """获取 agent 对应的 MuJoCo body id"""
        if hasattr(agent, 'mj_body_name') and agent.mj_body_name:
            return self.mjModel.body(agent.mj_body_name).id
        elif hasattr(agent, 'mj_body_id'):
            return agent.mj_body_id
        else:
            # 默认返回 0 或抛出错误
            raise ValueError(f"Agent {agent} has no MuJoCo body mapping")

    def destroy(self):
        """Clean up resources."""
        if self.viewer is not None:
            print("self.viewer closed")
            self.viewer.close()
            self.viewer = None
        super().destroy()