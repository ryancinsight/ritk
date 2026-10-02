use ritk_spatial::Direction;

pub(crate) fn reverse_axis_order<T>([first, middle, last]: [T; 3]) -> [T; 3] {
    [last, middle, first]
}

pub(crate) fn reverse_direction_axes(direction: Direction<3>) -> Direction<3> {
    let [first, middle, last] = direction.axis_directions_array();
    Direction::from_columns([last, middle, first])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_spatial::Vector;

    #[test]
    fn axis_order_reversal_preserves_middle_axis_and_moves_endpoints() {
        assert_eq!(reverse_axis_order([2, 5, 9]), [9, 5, 2]);
    }

    #[test]
    fn direction_reversal_reorders_columns_without_changing_rows() {
        let direction = Direction::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        assert_eq!(
            reverse_direction_axes(direction),
            Direction::from_columns([
                Vector::new([3.0, 6.0, 9.0]),
                Vector::new([2.0, 5.0, 8.0]),
                Vector::new([1.0, 4.0, 7.0]),
            ])
        );
    }
}
