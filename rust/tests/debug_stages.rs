use isophote_core::pipeline::*;

fn read_npy(path: &str) -> Vec<u8> {
    let buf = std::fs::read(path).unwrap();
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    buf[10 + hlen..].to_vec()
}

#[test]
#[allow(unused)]
fn debug_stages() {
    let data = read_npy("tests/in_bgr.npy");
    let h = 180usize; let w = 135usize;
    // Lab conversion itself
    let lab = bgr_to_lab(&data, w, h);
    println!("rust lab px(90,0) BGR-ish: {:?}", &lab[90 * w * 3..90 * w * 3 + 9]);
    let kuw = kuwahara(&data, w, h, 6);
    println!("rust kuwahara row90: {:?}", &kuw[90 * w * 3..90 * w * 3 + 9]);
    let _ = std::io::Write::flush(&mut std::io::stdout());
}
