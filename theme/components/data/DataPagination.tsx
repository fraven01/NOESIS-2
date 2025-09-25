import * as React from "react";
import { Pagination, PaginationProps } from "../ui/Pagination";

/**
 * Wrapper around Pagination for data lists.
 * @example
 * <DataPagination current={1} total={5} onChange={setPage} />
 */
export const DataPagination: React.FC<PaginationProps> = (props) => (
  <Pagination {...props} />
);
