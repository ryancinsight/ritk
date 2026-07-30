/// Two-row reconstructed-sample workspace shared by JPEG-LS encode and decode.
///
/// JPEG-LS prediction is causal: a sample depends on the previous row and the
/// reconstructed prefix of the current row. Keeping more than these two rows
/// does not affect the result.
pub(super) struct ReconstructionRows {
    samples: Box<[i32]>,
    columns: usize,
    previous_offset: usize,
    current_offset: usize,
    previous_line_left_guard: i32,
}

impl ReconstructionRows {
    pub(super) fn new(columns: usize) -> Self {
        let sample_count = columns
            .checked_mul(2)
            .expect("invariant: validated JPEG-LS columns fit two-row workspace");
        Self {
            samples: vec![0; sample_count].into_boxed_slice(),
            columns,
            previous_offset: 0,
            current_offset: columns,
            previous_line_left_guard: 0,
        }
    }

    #[inline(always)]
    pub(super) fn current_line_left_guard(&self) -> i32 {
        self.previous()[0]
    }

    #[inline(always)]
    pub(super) fn neighborhood(
        &self,
        column: usize,
        current_line_left_guard: i32,
    ) -> (i32, i32, i32, i32) {
        let previous = self.previous();
        let current = self.current();
        let left = if column > 0 {
            current[column - 1]
        } else {
            current_line_left_guard
        };
        let above = previous[column];
        let above_left = if column > 0 {
            previous[column - 1]
        } else {
            self.previous_line_left_guard
        };
        let above_right = if column + 1 < self.columns {
            previous[column + 1]
        } else {
            previous[column]
        };
        (left, above, above_left, above_right)
    }

    #[inline(always)]
    pub(super) fn interruption_neighbors(&self, column: usize) -> (i32, i32) {
        let above = self.previous()[column.min(self.columns - 1)];
        let left = if column > 0 {
            self.current()[column - 1]
        } else {
            above
        };
        (above, left)
    }

    #[inline(always)]
    pub(super) fn set_current(&mut self, column: usize, value: i32) {
        let offset = self.current_offset + column;
        self.samples[offset] = value;
    }

    pub(super) fn fill_current(&mut self, start: usize, length: usize, value: i32) {
        let end = start
            .checked_add(length)
            .expect("invariant: JPEG-LS run length fits the current row");
        self.current_mut()[start..end].fill(value);
    }

    pub(super) fn current(&self) -> &[i32] {
        &self.samples[self.current_offset..self.current_offset + self.columns]
    }

    pub(super) fn finish_row(&mut self, current_line_left_guard: i32) {
        self.previous_line_left_guard = current_line_left_guard;
        std::mem::swap(&mut self.previous_offset, &mut self.current_offset);
    }

    fn previous(&self) -> &[i32] {
        &self.samples[self.previous_offset..self.previous_offset + self.columns]
    }

    fn current_mut(&mut self) -> &mut [i32] {
        &mut self.samples[self.current_offset..self.current_offset + self.columns]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_stays_two_rows_after_rotation() {
        let mut rows = ReconstructionRows::new(512);
        assert_eq!(rows.samples.len(), 1_024);
        rows.set_current(0, 17);
        rows.finish_row(0);
        assert_eq!(rows.current_line_left_guard(), 17);
        assert_eq!(rows.samples.len() * std::mem::size_of::<i32>(), 4_096);
    }
}
