from silver_run import GPUDevice, GPUResourceScheduler, TaskResourceRequest


def test_gpu_scheduler_balances_percentages_and_memory():
    scheduler = GPUResourceScheduler([GPUDevice("0", 16000), GPUDevice("1", 16000)])
    plan = scheduler.plan([
        TaskResourceRequest("encoder", gpu_percent=75, memory_mb=8000),
        TaskResourceRequest("head", gpu_percent=75, memory_mb=8000),
        TaskResourceRequest("overflow", gpu_percent=75, memory_mb=8000),
    ])
    assert len(plan.allocations) == 2
    assert plan.rejected == ("overflow",)
    assert plan.for_task("encoder").environment()["SILVER_GPU_PERCENT"] == "75"
