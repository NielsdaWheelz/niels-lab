export type LedgerEvent = {
  date: string // YYYY-MM-DD
  kind: 'train' | 'read' | 'ship' | 'write'
  text: string // logbook shorthand: "SQ 5×5 @365 · Braudel 41pp"
  failed?: { lesson: string } // renders struck-through + lesson
  href?: string
}

export interface LedgerSource {
  get(): Promise<LedgerEvent[]>
}
