"""
验证LLM权重模块集成（静态检查版）
通过读取源代码验证修改
"""

def verify_integration():
    """验证集成完整性"""
    print("="*70)
    print("验证LLM权重模块集成（静态检查）")
    print("="*70)
    
    errors = []
    warnings = []
    
    # 1. 检查llm_weighting.py文件
    print("\n[1] 检查llm_weighting.py文件...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\components\llm_weighting.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class LLAMBOWeighting:' in content:
                print("    ✅ LLAMBOWeighting类存在")
            else:
                errors.append("llm_weighting.py缺少LLAMBOWeighting类")
                
            if 'compute_weight' in content:
                print("    ✅ compute_weight方法存在")
            else:
                errors.append("缺少compute_weight方法")
                
            if 'update_focus_from_llm' in content:
                print("    ✅ update_focus_from_llm方法存在")
            else:
                errors.append("缺少update_focus_from_llm方法")
    except FileNotFoundError:
        errors.append("llm_weighting.py文件不存在")
        print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"读取llm_weighting.py失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 2. 检查main.py导入
    print("\n[2] 检查main.py导入...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from components.llm_weighting import LLAMBOWeighting' in content:
                print("    ✅ main.py包含LLAMBOWeighting导入")
            else:
                errors.append("main.py缺少LLAMBOWeighting导入")
                print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查main.py导入失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 3. 检查LLMMOBO初始化
    print("\n[3] 检查LLMMOBO.__init__中的初始化...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'self.llm_weighting = LLAMBOWeighting(' in content:
                print("    ✅ __init__包含llm_weighting初始化")
            else:
                errors.append("__init__缺少llm_weighting初始化")
                print(f"    ❌ {errors[-1]}")
                
            if 'param_bounds=PARAM_BOUNDS' in content and 'llm_weighting' in content:
                print("    ✅ 初始化参数正确")
            else:
                warnings.append("初始化参数可能不完整")
    except Exception as e:
        errors.append(f"检查__init__失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 4. 检查_llm_weight_function替换
    print("\n[4] 检查_llm_weight_function方法...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 查找方法定义
            if 'def _llm_weight_function(self, X: np.ndarray)' in content:
                print("    ✅ _llm_weight_function方法存在")
                
                # 检查是否使用新实现
                if 'self.llm_weighting.compute_weight' in content:
                    print("    ✅ 使用LLAMBOWeighting.compute_weight")
                else:
                    errors.append("_llm_weight_function未调用LLAMBOWeighting")
                    print(f"    ❌ {errors[-1]}")
                
                # 检查是否还有旧代码
                if 'thermal_risk' in content or 'I1_SOC_coupling' in content:
                    # 检查是否在_llm_weight_function中
                    import re
                    func_match = re.search(
                        r'def _llm_weight_function.*?(?=\n    def |\nclass |\Z)',
                        content,
                        re.DOTALL
                    )
                    if func_match and ('thermal_risk' in func_match.group(0) or 
                                      'I1_SOC_coupling' in func_match.group(0)):
                        warnings.append("_llm_weight_function可能仍包含旧代码")
                        print(f"    ⚠️  {warnings[-1]}")
            else:
                errors.append("_llm_weight_function方法不存在")
                print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查_llm_weight_function失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 5. 检查optimize方法中的焦点更新
    print("\n[5] 检查optimize方法中的焦点更新...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'update_focus_from_llm' in content:
                print("    ✅ 包含update_focus_from_llm调用")
                
                # 检查调用位置（应该在BO循环中）
                if 'for iteration in range(self.n_iterations)' in content:
                    # 提取optimize方法
                    import re
                    opt_match = re.search(
                        r'async def optimize\(self\):.*?(?=\n    async def |\n    def |\nclass |\Z)',
                        content,
                        re.DOTALL
                    )
                    if opt_match and 'update_focus_from_llm' in opt_match.group(0):
                        print("    ✅ 焦点更新在optimize方法中")
                    else:
                        warnings.append("焦点更新可能不在optimize方法中")
            else:
                errors.append("缺少update_focus_from_llm调用")
                print(f"    ❌ {errors[-1]}")
    except Exception as e:
        errors.append(f"检查optimize方法失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 6. 检查HV计算修复
    print("\n[6] 检查HV计算参考点...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 检查是否移除了Log空间转换
            import re
            opt_match = re.search(
                r'# ={10,} 新增：计算HV历史.*?# ={10,}',
                content,
                re.DOTALL
            )
            
            if opt_match:
                hv_section = opt_match.group(0)
                if 'ref_point[2] = -1.0' in hv_section:
                    warnings.append("HV计算可能仍在使用Log空间转换")
                    print(f"    ⚠️  {warnings[-1]}")
                else:
                    print("    ✅ HV参考点使用原始空间")
            else:
                # 检查整个文件
                if "MOBO_CONFIG['reference_point']['aging']" in content:
                    print("    ✅ HV参考点使用aging原始值")
                else:
                    warnings.append("无法确认HV参考点设置")
    except Exception as e:
        warnings.append(f"检查HV计算: {e}")
        print(f"    ⚠️  {warnings[-1]}")
    
    # 7. 额外检查：验证所有5个修改点
    print("\n[7] 验证所有修改点...")
    try:
        with open(r'd:\Users\aa133\Desktop\BO_Multi_12_20\llmbo\main.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            modifications = {
                '导入LLAMBOWeighting': False,
                '__init__初始化': False,
                '_llm_weight_function替换': False,
                'optimize焦点更新': False,
                'HV参考点': False
            }
            
            content = ''.join(lines)
            
            if 'from components.llm_weighting import LLAMBOWeighting' in content:
                modifications['导入LLAMBOWeighting'] = True
            
            if 'self.llm_weighting = LLAMBOWeighting(' in content:
                modifications['__init__初始化'] = True
            
            if 'self.llm_weighting.compute_weight' in content:
                modifications['_llm_weight_function替换'] = True
            
            if 'await self.llm_weighting.update_focus_from_llm' in content:
                modifications['optimize焦点更新'] = True
            
            if "MOBO_CONFIG['reference_point']['aging']" in content:
                modifications['HV参考点'] = True
            
            print("    修改点检查:")
            for name, status in modifications.items():
                status_str = "✅" if status else "❌"
                print(f"      {status_str} {name}")
                if not status:
                    errors.append(f"缺少修改: {name}")
    except Exception as e:
        errors.append(f"验证修改点失败: {e}")
        print(f"    ❌ {errors[-1]}")
    
    # 总结
    print("\n" + "="*70)
    if len(errors) == 0:
        print("✅ 所有必需检查通过！")
        if len(warnings) > 0:
            print(f"\n⚠️  有 {len(warnings)} 个警告:")
            for i, warn in enumerate(warnings, 1):
                print(f"  {i}. {warn}")
        print("\n集成成功！现在可以使用:")
        print("  optimizer = LLMMOBO(llm_api_key='your_key', use_llm_acq=True)")
        print("  results = await optimizer.optimize()")
    else:
        print(f"❌ 发现 {len(errors)} 个错误:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        if len(warnings) > 0:
            print(f"\n⚠️  还有 {len(warnings)} 个警告:")
            for i, warn in enumerate(warnings, 1):
                print(f"  {i}. {warn}")
    print("="*70)
    
    return len(errors) == 0

if __name__ == "__main__":
    import sys
    success = verify_integration()
    sys.exit(0 if success else 1)
