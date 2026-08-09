import { log } from '@/content/log'
import type { LedgerSource } from './types'

export const manual: LedgerSource = {
  async get() {
    return log
  },
}
