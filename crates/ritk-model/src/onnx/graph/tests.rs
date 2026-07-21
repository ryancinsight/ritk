use super::*;

#[test]
fn test_graph_creation() {
    let graph = OnnxGraph::new("test_model".to_string());
    assert_eq!(graph.name, "test_model");
    assert!(graph.inputs.is_empty());
    assert!(graph.outputs.is_empty());
    assert!(graph.nodes.is_empty());
}

#[test]
fn test_node_attribute_extraction() {
    let mut node = OnnxNode::new("conv1".to_string(), "Conv".to_string());
    node.attributes
        .insert("kernel_size".to_string(), OnnxAttribute::Ints(vec![3, 3]));
    node.attributes
        .insert("stride".to_string(), OnnxAttribute::Int(1));
    node.attributes
        .insert("epsilon".to_string(), OnnxAttribute::Float(1e-5));

    let kernel_size: Vec<i64> = node.get_attr("kernel_size").expect("infallible: validated precondition");
    assert_eq!(kernel_size, vec![3, 3]);

    let stride: i64 = node.get_attr("stride").expect("infallible: validated precondition");
    assert_eq!(stride, 1);

    let epsilon: f32 = node.get_attr("epsilon").expect("infallible: validated precondition");
    assert!((epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_optional_attribute() {
    let node = OnnxNode::new("relu1".to_string(), "Relu".to_string());

    let alpha: Option<f32> = node.get_attr_opt("alpha").expect("infallible: validated precondition");
    assert!(alpha.is_none());

    let default_alpha = node.get_attr_or("alpha", 0.0f32).expect("infallible: validated precondition");
    assert!((default_alpha - 0.0).abs() < 1e-10);
}

#[test]
fn test_tensor_creation() {
    let tensor = OnnxTensor::new(
        "weight".to_string(),
        vec![64, 3, 7, 7],
        OnnxElementType::Float,
    );
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.num_elements(), 64 * 3 * 7 * 7);
    assert_eq!(tensor.byte_size(), 64 * 3 * 7 * 7 * 4);
}

#[test]
fn test_value_info() {
    let info = OnnxValueInfo::new(OnnxElementType::Float, vec![1, 3, 224, 224]);
    assert!(info.is_static());
    assert_eq!(info.num_elements(), Some(3 * 224 * 224));

    let dynamic_info = OnnxValueInfo::new(OnnxElementType::Float, vec![-1, 3, 224, 224]);
    assert!(!dynamic_info.is_static());
    assert!(dynamic_info.num_elements().is_none());
}

#[test]
fn test_graph_validation() {
    let mut graph = OnnxGraph::new("test".to_string());

    // Add input
    graph.inputs.push(OnnxValue::new(
        "input".to_string(),
        OnnxElementType::Float,
        vec![1, 3, 32, 32],
    ));

    // Add initializer
    let weight = OnnxValue::new(
        "weight".to_string(),
        OnnxElementType::Float,
        vec![16, 3, 3, 3],
    );
    graph.initializers.insert("weight".to_string(), weight);

    // Add node
    let mut conv = OnnxNode::new("conv1".to_string(), "Conv".to_string());
    conv.inputs = vec!["input".to_string(), "weight".to_string()];
    conv.outputs = vec!["output".to_string()];
    graph.nodes.push(conv);

    // Add output
    graph.outputs.push(OnnxValue::new(
        "output".to_string(),
        OnnxElementType::Float,
        vec![1, 16, 30, 30],
    ));

    assert!(graph.validate().is_ok());
}

#[test]
fn test_element_type_properties() {
    assert!(OnnxElementType::Float.is_float());
    assert!(!OnnxElementType::Float.is_int());
    assert!(OnnxElementType::Int32.is_int());
    assert!(!OnnxElementType::Int32.is_float());
    assert_eq!(OnnxElementType::Float.element_size(), 4);
    assert_eq!(OnnxElementType::Int64.element_size(), 8);
    assert_eq!(OnnxElementType::Float16.element_size(), 2);
}
