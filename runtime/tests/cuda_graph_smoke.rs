//! Smoke test for CUDA graph capture + replay. Records a single
//! kernel launch into a graph, replays it many times, and verifies
//! the output is unchanged.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_i32, aether_dev_h2d_i32,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_dev_graph_begin, aether_dev_graph_end,
    aether_dev_graph_launch, aether_dev_graph_destroy,
};

#[test]
#[ignore]
fn graph_capture_replay_rope() {
    unsafe {
        aether_dev_init();
        const SEQ: c_int = 1;
        const N_HEADS: c_int = 28;
        const HEAD_DIM: c_int = 128;
        const BASE: f32 = 1_000_000.0;
        let n = (SEQ * N_HEADS * HEAD_DIM) as usize;

        let x_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let d_x = aether_dev_alloc_f32(n as c_int);
        let d_step = aether_dev_alloc_i32(4);

        // ---- Reference run (no graph), pos=3 ----
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x, n as c_int);
        let step_ref = [3i32, 0, 0, 0];
        aether_dev_h2d_i32(step_ref.as_ptr() as i64, d_step, 4);
        assert_eq!(0, aether_op_rope_apply_devarg_f32_cuda(d_x, SEQ, N_HEADS, HEAD_DIM, BASE, d_step));
        aether_dev_sync();
        let mut ref_out = vec![0.0f32; n];
        aether_dev_d2h_f32(d_x, ref_out.as_mut_ptr() as i64, n as c_int);

        // ---- Capture into graph (with pos=3 in step_args), then replay ----
        // Reset device buffer first.
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x, n as c_int);
        aether_dev_h2d_i32(step_ref.as_ptr() as i64, d_step, 4);
        aether_dev_sync();

        assert_eq!(0, aether_dev_graph_begin(), "graph_begin failed");
        assert_eq!(0, aether_op_rope_apply_devarg_f32_cuda(d_x, SEQ, N_HEADS, HEAD_DIM, BASE, d_step));
        assert_eq!(0, aether_dev_graph_end(), "graph_end failed");

        // The capture itself ALSO records the side effect of the kernel,
        // so d_x now holds the result of one rope application.
        // Reset to fresh input for the graph launch test.
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x, n as c_int);
        aether_dev_sync();

        assert_eq!(0, aether_dev_graph_launch(), "graph_launch failed");
        aether_dev_sync();

        let mut graph_out = vec![0.0f32; n];
        aether_dev_d2h_f32(d_x, graph_out.as_mut_ptr() as i64, n as c_int);

        let mut max_diff = 0.0f32;
        let mut bad = 0;
        for i in 0..n {
            let d = (ref_out[i] - graph_out[i]).abs();
            if d > max_diff { max_diff = d; }
            if d > 0.0 { bad += 1; }
        }
        eprintln!("[graph rope replay] max_diff={:.3e} bad={}/{}", max_diff, bad, n);
        eprintln!("  ref[0..4]   = {:?}", &ref_out[..4]);
        eprintln!("  graph[0..4] = {:?}", &graph_out[..4]);
        assert_eq!(bad, 0, "graph replay produces different output");

        // ---- Now update step_args to pos=5, replay graph, verify it
        //      uses the NEW pos (devarg reads from device memory) ----
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x, n as c_int);
        let step_new = [5i32, 0, 0, 0];
        aether_dev_h2d_i32(step_new.as_ptr() as i64, d_step, 4);
        aether_dev_sync();
        assert_eq!(0, aether_dev_graph_launch());
        aether_dev_sync();
        let mut pos5_out = vec![0.0f32; n];
        aether_dev_d2h_f32(d_x, pos5_out.as_mut_ptr() as i64, n as c_int);

        // Compute the reference for pos=5 directly.
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x, n as c_int);
        let step_ref5 = [5i32, 0, 0, 0];
        aether_dev_h2d_i32(step_ref5.as_ptr() as i64, d_step, 4);
        assert_eq!(0, aether_op_rope_apply_devarg_f32_cuda(d_x, SEQ, N_HEADS, HEAD_DIM, BASE, d_step));
        aether_dev_sync();
        let mut ref5 = vec![0.0f32; n];
        aether_dev_d2h_f32(d_x, ref5.as_mut_ptr() as i64, n as c_int);

        let mut bad5 = 0;
        for i in 0..n {
            if (ref5[i] - pos5_out[i]).abs() > 0.0 { bad5 += 1; }
        }
        eprintln!("[graph rope replay pos=5] bad={}/{}", bad5, n);
        assert_eq!(bad5, 0, "graph replay does not pick up updated step_args");

        aether_dev_graph_destroy();
        aether_dev_free_f32(d_x);
    }
}
