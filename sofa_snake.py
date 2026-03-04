"""
sofa_snake.py
=============
Tendon-driven continuum soft robot (FEM volumetric) in SofaPython3 (v25.12).
Demonstrates Model-Free Adaptive Jacobian steering with catastrophic
tendon failure at T=3.0s and online re-adaptation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimator import JacobianEstimator   

import Sofa
import Sofa.Core
import Sofa.Simulation

# ===========================================================================
# GEOMETRY HELPERS 
# ===========================================================================

def generate_cylinder_mesh(output_path: str, length: float = 0.12, radius: float = 0.012, lc: float = 0.006) -> str:
    try:
        import gmsh
    except ImportError:
        raise RuntimeError("gmsh Python package not found. Install with: pip install gmsh")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")
    gmsh.model.occ.addCylinder(-length / 2, 0, 0, length, 0, 0, radius)               
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)  
    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)
    gmsh.finalize()
    return output_path

def cable_anchor_points(radius: float, length: float, offset_r: float, n_points: int = 8) -> list:
    xs = np.linspace(-length / 2 + 0.002, length / 2 - 0.002, n_points)
    dy, dz = offset_r
    return [(float(x), dy, dz) for x in xs]


# ===========================================================================
# ADAPTIVE CONTROLLER
# ===========================================================================

class AdaptiveController(Sofa.Core.Controller):
    def __init__(self, snake_node, cable_left, cable_right, cable_top, cable_bottom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snake_node   = snake_node
        self.cable_left   = cable_left
        self.cable_right  = cable_right
        self.cable_top    = cable_top
        self.cable_bottom = cable_bottom

        self.dt           = 0.02          
        self.t            = 0.0
        self.gait_freq    = 1.5           
        self.gait_amp     = 0.008         

        self.K_p          = 3.0           
        self.bias         = 0.0           
        self.bias_max     = 0.006         

        self.estimator    = JacobianEstimator(alpha=0.6)
        self.tendon_severed = False
        self.T_damage       = 3.0         
        self.log          = []            

    def _get_z_head(self) -> float:
        mo  = self.snake_node.getMechanicalState()
        pos = np.array(mo.position.value)   
        return float(pos[-1, 2])

    def _set_cable_pull(self, cable_node, pull_distance: float):
        pull_distance = max(0.0, pull_distance)   
        try:
            cable_node.CableConstraint.value.value = [pull_distance]
        except AttributeError:
            cable_node.CableConstraint.cableLength.value = pull_distance

    def onAnimateBeginEvent(self, event):
        self.t += self.dt

        # ── ANISOTROPIC FRICTION ──
        mo  = self.snake_node.getMechanicalState()
        pos = np.array(mo.position.value)          
        vel = np.array(mo.velocity.value)          

        body_vec = pos[-1] - pos[0]
        body_len = np.linalg.norm(body_vec)
        if body_len > 1e-9:                        
            t_hat = body_vec / body_len            
            proj_scalar = vel @ t_hat              
            v_fwd = proj_scalar[:, np.newaxis] * t_hat[np.newaxis, :]  
            v_lat = vel - v_fwd                    
            vel_damped = vel - (0.01 * v_fwd + 0.8 * v_lat)  
            mo.velocity.value = vel_damped.tolist()

        # ── JACOBIAN UPDATE ──
        z_head = self._get_z_head()
        u_vec = np.array([self.bias,  0.0])
        x_vec = np.array([z_head,     0.0])
        J_est = self.estimator.update(u_vec, x_vec, time_s=self.t)

        J11 = J_est[0, 0]
        if abs(J11) < 1e-6:
            J11 = 1e-6 * np.sign(J11 + 1e-12)

        # ── CONTROL LAW & DAMAGE ──
        x_err   = z_head              
        delta_u = -self.K_p * x_err / J11
        self.bias = float(np.clip(self.bias + delta_u * self.dt, -self.bias_max, self.bias_max))

        if self.t >= self.T_damage and not self.tendon_severed:
            self.tendon_severed = True
            print(f"\n[T={self.t:.2f}s]  *** RIGHT TENDON SEVERED ***\n")

        # ── GAIT KINEMATICS ──
        phase   = 2.0 * np.pi * self.gait_freq * self.t
        base    = self.gait_amp * (0.5 + 0.5 * np.sin(phase))   

        pull_left  = float(np.clip(base + self.bias,  0.0, self.gait_amp * 2))
        pull_right_gait = self.gait_amp * (0.5 - 0.5 * np.sin(phase))
        pull_right = 0.0 if self.tendon_severed else float(np.clip(pull_right_gait - self.bias, 0.0, self.gait_amp * 2))

        pull_top    = self.gait_amp * 0.3 * abs(np.sin(phase))
        pull_bottom = pull_top  

        self._set_cable_pull(self.cable_left,   pull_left)
        self._set_cable_pull(self.cable_right,  pull_right)
        self._set_cable_pull(self.cable_top,    pull_top)
        self._set_cable_pull(self.cable_bottom, pull_bottom)

        self.log.append((self.t, z_head, self.bias, J11, pull_left, pull_right))
        
        # Only print to console every 50 frames to save I/O time
       # I/O Optimization: Print status at 50Hz, not 1000Hz.
        if len(self.log) % 50 == 0:
            status = "SEVERED" if self.tendon_severed else "NOMINAL"
            print(f"t={self.t:.2f}s | Z-Drift: {z_head*1000:+.2f}mm | Bias: {self.bias*1000:+.2f}mm | {status}")
            self.save_log("snake_log.csv") # <-- ADD THIS LINE: Flushes RAM to disk every 1 second of simulation time

    def save_log(self, path: str = "snake_log.csv"):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "z_head_m", "bias_m", "J11", "pull_left_m", "pull_right_m"])
            w.writerows(self.log)
        print(f"Log saved to {path}")


# ===========================================================================
# SCENE CREATION
# ===========================================================================

def createScene(rootNode):
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    mesh_path    = os.path.join(script_dir, "cylinder.msh")

    if not os.path.exists(mesh_path):
        generate_cylinder_mesh(mesh_path, length=0.12, radius=0.012, lc=0.005)

    L      = 0.12    
    R      = 0.012   
    r_off  = 0.008   

    rootNode.dt = 0.02
    rootNode.gravity = [0.0, -9.81, 0.0]
    rootNode.name = "rootNode"

    plugins = [
        "Sofa.Component.AnimationLoop", "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Model", "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Collision.Detection.Algorithm", "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry", "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.IO.Mesh", "Sofa.Component.LinearSolver.Iterative", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mass", "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic", "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic", "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Container.Constant", "Sofa.GL.Component.Rendering3D", "SoftRobots"
    ]
    for p in plugins:
        rootNode.addObject("RequiredPlugin", name=p, printLog=False)

    rootNode.addObject("FreeMotionAnimationLoop", name="AnimationLoop", parallelODESolving=False, printLog=False)
    rootNode.addObject("BlockGaussSeidelConstraintSolver", name="ConstraintSolver", maxIterations=200, tolerance=1e-7, printLog=False)

    # ── OPTIMIZED & MUTED COLLISION PIPELINE ──
    rootNode.addObject("CollisionPipeline",  name="CollisionPipeline", printLog=False)
    rootNode.addObject("BruteForceBroadPhase", printLog=False)
    rootNode.addObject("BVHNarrowPhase", printLog=False)
    rootNode.addObject("DiscreteIntersection", printLog=False)
    rootNode.addObject("CollisionResponse", name="ContactManager", response="FrictionContactConstraint", responseParams="mu=0.7", printLog=False)
    rootNode.addObject("LocalMinDistance", name="Proximity", alarmDistance=0.003, contactDistance=0.001, angleCone=0.01, printLog=False)

    # ── FLOOR NODE (TRIANGLES ONLY) ──
    floorNode = rootNode.addChild("floorNode")
    floorNode.addObject("MeshTopology", name="FloorTopo", position=[[-1, -0.013, -1], [ 1, -0.013, -1], [ 1, -0.013,  1], [-1, -0.013,  1]], triangles=[[0, 1, 2], [0, 2, 3]])
    floorNode.addObject("MechanicalObject", name="FloorMO", template="Vec3d")
    floorNode.addObject("TriangleCollisionModel", name="FloorTriCM", simulated=False, moving=False)

    # ── SNAKE NODE ──
    snakeNode = rootNode.addChild("snakeNode")
    snakeNode.addObject("EulerImplicitSolver", name="ODE", rayleighStiffness=0.1, rayleighMass=0.1, printLog=False)
    
    # Adding compressed matrix template to satisfy the SparseLDLSolver warning
    snakeNode.addObject("SparseLDLSolver", name="LinearSolver", template="CompressedRowSparseMatrixd", printLog=False)

    snakeNode.addObject("MeshGmshLoader", name="MeshLoader", filename=mesh_path, rotation=[0, 0, 0], translation=[0, 0, 0])
    snakeNode.addObject("TetrahedronSetTopologyContainer", name="Topo", src="@MeshLoader")
    snakeNode.addObject("TetrahedronSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
    snakeNode.addObject("TetrahedronSetTopologyModifier", name="TopoModifier")
    snakeNode.addObject("MechanicalObject", name="MO", template="Vec3d", src="@MeshLoader")

    mass = 1100.0 * (np.pi * R**2 * L)          
    snakeNode.addObject("UniformMass", name="Mass", totalMass=mass)
    snakeNode.addObject("TetrahedronFEMForceField", name="FEM", template="Vec3d", method="large", poissonRatio=0.45, youngModulus=1e5, printLog=False)
    snakeNode.addObject("LinearSolverConstraintCorrection", name="ConstraintCorrection", printLog=False)

    # ── SNAKE COLLISION SURFACE (TRIANGLES ONLY) ──
    collNode = snakeNode.addChild("CollisionSurface")
    collNode.addObject("MeshGmshLoader", name="SurfLoader", filename=mesh_path)
    collNode.addObject("MeshTopology", name="SurfTopo", src="@SurfLoader")
    collNode.addObject("MechanicalObject", name="SurfMO", template="Vec3d", src="@SurfLoader")
    collNode.addObject("TriangleCollisionModel", name="TriCM", selfCollision=False)
    collNode.addObject("BarycentricMapping", name="BaryMap", input="@../MO", output="@SurfMO")

    cable_specs = {"cableLeft": (0.0, r_off), "cableRight": (0.0, -r_off), "cableTop": (r_off, 0.0), "cableBottom": (-r_off, 0.0)}
    cable_nodes = {}
    for cname, (dy, dz) in cable_specs.items():
        anchors = cable_anchor_points(R, L, (dy, dz), n_points=10)
        anchor_str = " ".join(f"{p[0]} {p[1]} {p[2]}" for p in anchors)
        indices    = list(range(len(anchors)))
        cNode = snakeNode.addChild(cname)
        cNode.addObject("MechanicalObject", name="CableMO", template="Vec3d", position=anchor_str)
        cNode.addObject("CableConstraint", name="CableConstraint", template="Vec3d", indices=indices, value=[0.0], valueType="displacement", hasPullPoint=False)
        cNode.addObject("BarycentricMapping", name="BaryMap", input="@../MO", output="@CableMO")
        cable_nodes[cname] = cNode

    visNode = snakeNode.addChild("Visual")
    visNode.addObject("MeshGmshLoader", name="VisLoader", filename=mesh_path)
    visNode.addObject("OglModel", name="OglModel", src="@VisLoader", color="0.3 0.7 0.4 0.9")
    visNode.addObject("BarycentricMapping", name="VisMap", input="@../MO", output="@OglModel")

    snakeNode.addObject(AdaptiveController(name="AdaptiveController", snake_node=snakeNode, cable_left=cable_nodes["cableLeft"], cable_right=cable_nodes["cableRight"], cable_top=cable_nodes["cableTop"], cable_bottom=cable_nodes["cableBottom"]))

    return rootNode

if __name__ == "__main__":
    import Sofa.Gui
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    root = Sofa.Core.Node("rootNode")
    createScene(root)
    Sofa.Simulation.init(root)
    for step in range(n_steps):
        Sofa.Simulation.animate(root, root.dt.value)
    root.snakeNode.AdaptiveController.save_log("snake_log.csv")