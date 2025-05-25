# TODO

Looking at this vidkompy video alignment benchmark log, here's a **TL;DR of execution times and performance** for different engines and settings:

## Performance Summary by Engine (Processing ~8-second videos)

### **🏆 Fastest: tunnel_full**
- **d10-w10**: 40.9s ⭐ (fastest overall)
- **d100-w10**: 41.6s 
- **d10-w100**: 1m33s
- **d100-w100**: 1m33s
- **Confidence**: 1.000 (perfect)

### **🥈 Second: tunnel_mask** 
- **d10-w10**: 45.8s
- **d100-w10**: 50.0s
- **d10-w100**: 4m8s ⚠️ (slowest overall)
- **d100-w100**: 2m17s
- **Confidence**: 1.000 (perfect)

### **🥉 Third: fast engine**
- **All configs**: ~52-54s (very consistent)
- **Confidence**: 0.300 ⚠️ (poor - fell back to direct mapping)

### **Fourth: precise/mask engines**
- **d10-w10**: 58s - 1m4s
- **d10-w100**: 1m4s - 1m8s  
- **d100 configs**: 1m19s - 1m22s
- **Confidence**: 0.606 (moderate)

## **Key Insights:**

1. **Window size impact**: Large windows (w100) dramatically slow down tunnel engines but barely affect others
2. **Drift interval**: Less impact than window size across all engines
3. **Quality vs Speed**: tunnel_full/tunnel_mask achieve perfect confidence (1.0) with reasonable speed
4. **Fast engine failed**: Despite the name, it couldn't find keyframe matches and used fallback mode


## TASKS: 

### [ ] Remove ineffective engines

- [ ] Completely remove the `fast` engine
- [ ] Completely remove the `precise` engine. 
- [ ] Completely remove the `mask` engine. 

Make sure to keep the code necessary for the `tunnel_` engines but remove all other code that is not necessary. 

### [ ] Rename the `tunnel_` engines

- [ ] Rename the `tunnel_full` engine to `full` and make it default.  
- [ ] Rename the `tunnel_mask` engine to `mask`.  

### [ ] Change the default values for draft interval and for window

- [ ] Make the draft interval default value to be 10
- [ ] Make the window default value to be 10. 

### [ ] Update docs

- [ ] Update the CHANGELOG.md file to reflect the changes. 
- [ ] Update the README.md file to reflect the changes. 
- [ ] Update the SPEC.md file to reflect the changes. 
- [ ] Update the PLAN.md file to reflect the changes. Remove all things already done, just keep things that are still to be done. 
- [ ] Update the TODO.md file to reflect the changes. Eliminate all things already done, just keep things that are still to be done. 

### [ ] Update tests

- [ ] Update the tests to reflect the changes. 
- [ ] Update the benchmark.sh file to reflect the changes. 
- [ ] Update the benchmark.sh file to reflect the changes. 

---