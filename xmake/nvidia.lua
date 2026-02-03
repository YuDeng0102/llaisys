target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    
    set_languages("cxx17")
    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart", "cublas")
    
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
        add_cflags("-fPIC")

    end
    
    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-utils", "llaisys-device-nvidia")
    
    set_languages("cxx17")
    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart", "cublas")
    
    
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
        add_cflags("-fPIC")
    end
    
    add_files("../src/ops/**/nvidia/*.cu")
    on_install(function (target) end)
target_end()
