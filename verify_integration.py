"""
验证LLM权重模块集成
快速检查所有修改是否正确应用
"""

import sys
sys.path.insert(0, r'd:\Users\aa133\Desktop\BO_Multi_12_20')

def verify_integration():
    """验证集成完整性"""
    print("="*70)
    print("验证LLM权重模块集成")
    print("="*70)
    
    errors = []
    
    # 1. 检查llm_weighting.py是否存在
    print("\n[1] 检查llm_weighting.py文件...")
    try:
        from llmbo.components.llm_weighting import LLAMBOWeighting
        print("    ✅ LLAMBOWeighting类导入成功")
    except Exception as e:
        errors.append(f"导入LLAMBOWeighting失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 2. 检查main.py导入
    print("\n[2] 检查main.py导入...")
    try:
        import llmbo.main as main_module
        if hasattr(main_module, 'LLAMBOWeighting'):
            print("    ✅ main.py已导入LLAMBOWeighting")
        else:
            # 检查源代码
            import inspect
            source = inspect.getsource(main_module)
            if 'from components.llm_weighting import LLAMBOWeighting' in source:
                print("    ✅ main.py包含正确的导入语句")
            else:
                errors.append("main.py缺少LLAMBOWeighting导入")
                print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查main.py失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 3. 检查LLMMOBO类初始化
    print("\n[3] 检查LLMMOBO初始化代码...")
    try:
        from llmbo.main import LLMMOBO
        import inspect
        source = inspect.getsource(LLMMOBO.__init__)
        
        if 'self.llm_weighting = LLAMBOWeighting(' in source:
            print("    ✅ __init__中包含LLM权重模块初始化")
        else:
            errors.append("__init__缺少llm_weighting初始化")
            print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查__init__失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 4. 检查_llm_weight_function方法
    print("\n[4] 检查_llm_weight_function方法...")
    try:
        from llmbo.main import LLMMOBO
        import inspect
        source = inspect.getsource(LLMMOBO._llm_weight_function)
        
        if 'self.llm_weighting.compute_weight' in source:
            print("    ✅ _llm_weight_function调用LLAMBOWeighting")
        else:
            errors.append("_llm_weight_function未使用新实现")
            print(f"    ❌ {errors[-1]}")
            
        # 检查是否还有旧的风险计算代码
        if 'thermal_risk' in source or 'I1_SOC_coupling' in source:
            errors.append("_llm_weight_function仍包含旧的风险计算代码")
            print(f"    ⚠️  警告: {errors[-1]}")
    except Exception as e:
        errors.append(f"检查_llm_weight_function失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 5. 检查optimize方法中的焦点更新
    print("\n[5] 检查optimize方法中的焦点更新...")
    try:
        from llmbo.main import LLMMOBO
        import inspect
        source = inspect.getsource(LLMMOBO.optimize)
        
        if 'update_focus_from_llm' in source:
            print("    ✅ optimize方法包含焦点更新调用")
        else:
            errors.append("optimize方法缺少焦点更新")
            print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查optimize方法失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 6. 检查HV计算修复
    print("\n[6] 检查HV计算参考点修复...")
    try:
        from llmbo.main import LLMMOBO
        import inspect
        source = inspect.getsource(LLMMOBO.optimize)
        
        # 检查是否移除了Log空间调整
        if 'ref_point[2] = -1.0' in source:
            errors.append("HV计算仍使用Log空间参考点（应已移除）")
            print(f"    ⚠️  警告: {errors[-1]}")
        else:
            print("    ✅ HV计算使用原始空间参考点")
    except Exception as e:
        errors.append(f"检查HV计算失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 总结
    print("\n" + "="*70)
    if len(errors) == 0:
        print("✅ 所有检查通过！LLM权重模块集成成功。")
        print("\n现在可以运行:")
        print("  optimizer = LLMMOBO(llm_api_key='your_key', use_llm_acq=True)")
        print("  results = await optimizer.optimize()")
    else:
        print(f"❌ 发现 {len(errors)} 个问题:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
    print("="*70)
    
    return len(errors) == 0

if __name__ == "__main__":
    success = verify_integration()
    sys.exit(0 if success else 1)
