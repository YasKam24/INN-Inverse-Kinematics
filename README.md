# Inverse Kinematics with Invertible Neural Network

This project implements **Inverse Kinematics (IK)** using an **Invertible Neural Network (INN)**. Given a desired end-effector (target) position of a robotic arm, the model learns to predict **all valid joint configurations** that can achieve that position. Additionally, users can modify the number of joints and any other parameters, This code also gives a visualization of all the arms. 

## Features

-  Bidirectional mapping between configuration space and task space.
-  Predicts **multiple valid solutions** for a given end-effector target.
-  Built on Invertible Neural Networks (INNs).
-  Supports multi-joint robotic arms.
-  Allows smooth configuration control even after solving IK.

---

## Background

In many IK scenarios, a single target position can correspond to **multiple joint configurations** due to the redundancy in robotic arms. Traditional IK solvers usually provide only one solution or rely on optimization tricks. INNs allow us to learn a **bijective mapping** between joint angles and end-effector positions, enabling us to **sample diverse and valid solutions** while preserving invertibility.

---


*This project was developed as part of the UMC203 course. You can also explore my teammate's implementation using INNs [here](https://github.com/IISc-UMC-203)*
