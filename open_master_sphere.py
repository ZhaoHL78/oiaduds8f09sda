import kikuchipy as kp

# 读取 kikuchipy 自带的小尺寸 Ni EBSD master pattern
# hemisphere="both" 更稳，适合完整球面显示
mp = kp.data.nickel_ebsd_master_pattern_small(hemisphere="both")

print(type(mp))
print("projection:", mp.projection)

# 打开 3D 菊池球窗口
# notebook=False 表示在独立窗口打开，而不是 notebook 内嵌
mp.plot_spherical(
    plotter_kwargs={"notebook": False}
)